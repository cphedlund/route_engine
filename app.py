# app.py
from __future__ import annotations

import os
import time
import json
import hmac
import hashlib
import base64
import difflib
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dotenv import load_dotenv
from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field
import osm_layers

load_dotenv(override=False)

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gpx_loader import load_routes_from_gpx_dir
from engine import Route, proximity_exhausted_notice, select_routes_with_relaxation_steps
import os
from fastapi import Header, HTTPException

# -----------------------------
# App + CORS
# -----------------------------
app = FastAPI(title="Route Selection Engine")

def validate_startup_config(environ=None) -> None:
    env = os.environ if environ is None else environ
    flag = str(env.get("REQUIRE_API_KEY", "0")).strip().lower() in ("1", "true", "yes", "on")
    if flag and not str(env.get("ROUTE_ENGINE_API_KEY", "")).strip():
        raise RuntimeError(
            "REQUIRE_API_KEY is enabled but ROUTE_ENGINE_API_KEY is missing or empty. "
            "Set ROUTE_ENGINE_API_KEY or unset REQUIRE_API_KEY."
        )


@app.on_event("startup")
async def startup_event():
    validate_startup_config()

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://lovable.dev",
    "https://www.lovable.dev",
    "https://atlasnav.lovable.app",
    "https://atlasai.lovable.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# API Key Auth
# -----------------------------
API_KEY = os.getenv("ROUTE_ENGINE_API_KEY", "")
DEV_MODE = os.getenv("DEV_MODE", "0") == "1"


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# -----------------------------
# Optional OpenAI LLM Translation
# -----------------------------
LLM_TRANSLATION_ENABLED = os.getenv("LLM_TRANSLATION_ENABLED", "0") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "6.0"))
OPENAI_MIN_CONFIDENCE = float(os.getenv("OPENAI_MIN_CONFIDENCE", "0.55"))

# Tiny in-memory cache: query -> (ts, payload)
_LLM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_LLM_CACHE_TTL = float(os.getenv("OPENAI_TRANSLATION_CACHE_TTL_SECONDS", "900"))  # 15 min
_LLM_CACHE_MAX = int(os.getenv("OPENAI_TRANSLATION_CACHE_MAX", "256"))


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    item = _LLM_CACHE.get(key)
    if not item:
        return None
    ts, payload = item
    if (now - ts) > _LLM_CACHE_TTL:
        _LLM_CACHE.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: Dict[str, Any]) -> None:
    if len(_LLM_CACHE) >= _LLM_CACHE_MAX:
        oldest_key = None
        oldest_ts = None
        for k, (ts, _) in _LLM_CACHE.items():
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
                oldest_key = k
        if oldest_key is not None:
            _LLM_CACHE.pop(oldest_key, None)
    _LLM_CACHE[key] = (time.time(), payload)


def _openai_client():
    # Lazy import: allows running without openai installed if LLM is disabled
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai SDK not installed. Run: pip install openai") from e
    return OpenAI()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_text(s: Optional[str]) -> str:
    return " ".join(str(s or "").strip().lower().split())


# -----------------------------
# Stateless session token
# -----------------------------
SESSION_SECRET = os.getenv("ROUTE_ENGINE_SESSION_SECRET", "")
if not SESSION_SECRET and DEV_MODE:
    SESSION_SECRET = "dev-session-secret"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _sign(payload_b64: str) -> str:
    if not SESSION_SECRET:
        raise HTTPException(status_code=500, detail="Missing ROUTE_ENGINE_SESSION_SECRET")
    sig = hmac.new(
        SESSION_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(sig)


def make_session_token(data: Dict[str, Any]) -> str:
    payload_json = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)
    sig_b64 = _sign(payload_b64)
    return f"{payload_b64}.{sig_b64}"


def read_session_token(token: str, max_age_seconds: int = 3600) -> Dict[str, Any]:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid session_id token")
    expected = _sign(payload_b64)
    if not hmac.compare_digest(expected, sig_b64):
        raise HTTPException(status_code=401, detail="Invalid session_id token")
    try:
        data = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session_id token")
    created_at = int(data.get("created_at", 0) or 0)
    now = int(time.time())
    if created_at <= 0 or (now - created_at) > max_age_seconds:
        raise HTTPException(status_code=401, detail="Session expired")
    return data


class LocationPref(BaseModel):
    lat: float
    lng: float
    radius_km: float = 20


class RangePref(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class Preferences(BaseModel):
    intent: Optional[Literal["run", "hike", "bike"]] = None
    location: Optional[LocationPref] = None
    distance_km: Optional[RangePref] = None
    elevation_gain_m: Optional[RangePref] = None
    surface: Optional[List[str]] = None
    avoid: Optional[List[str]] = None
    priorities: Optional[Dict[str, float]] = None  # e.g. {"views":0.8,"shade":0.5}
    must: Optional[Dict[str, bool]] = None          # e.g. {"loop": True}
    lat: Optional[float] = None  # user coordinates for live proximity
    lng: Optional[float] = None


class StartSearchBody(BaseModel):
    query: str
    preferences: Preferences = Field(default_factory=Preferences)
    batch_size: int = 3
    min_conformity: int = 85
    session_id: Optional[str] = None
    new_search: bool = False


# -----------------------------
# Models
# -----------------------------
class RouteIn(BaseModel):
    route_id: str
    name: str
    location: str
    distance_miles: float
    elevation_gain: float
    surface_type: str
    shade_pct: float
    scenic_likelihood: float
    proximity_miles: float
    route_type: str
    popularity: float = 0.0
    difficulty: str = "unknown"
    technicality: str = "unknown"
    max_grade_pct: float = 0.0
    avg_grade_pct: float = 0.0
    start_lat: Optional[float] = None
    start_lng: Optional[float] = None
    # ----------------------------------------------------------------
    # OSM-derived fields (must mirror Route dataclass defaults)
    # ----------------------------------------------------------------
    osm_surface: str = "unknown"
    osm_highway: str = "unknown"
    osm_smoothness: str = ""
    osm_bicycle_legal: bool = True
    osm_bicycle_no_pct: float = 0.0
    osm_horse_legal: bool = False
    osm_dog_allowed: Optional[bool] = None
    osm_technicality: float = 0.0
    osm_sac_scale: float = 0.0
    osm_mtb_scale: float = 0.0
    osm_has_trailhead_parking: bool = False
    osm_free_parking: bool = False
    osm_scenic_poi_count: int = 0
    osm_water_count: int = 0
    osm_drinking_water_count: int = 0
    osm_restroom_count: int = 0
    osm_shade_pct: int = 0
    osm_park_name: str = ""
    osm_park_operator: str = ""
    osm_park_dog_policy: str = ""
    osm_park_fee: str = ""
    osm_park_assignment: str = ""
    osm_park_overlap_pct: float = 0.0
    osm_picnic_count: int = 0
    osm_camping_count: int = 0
    def to_engine_route(self) -> Route:
        d = self.model_dump()
        start_lat = d.pop("start_lat", None)
        start_lng = d.pop("start_lng", None)
        if start_lat is not None and start_lng is not None:
            d["_start_point"] = (float(start_lat), float(start_lng))
        return Route(**d)


class StartSearchRequest(BaseModel):
    query: Optional[str] = None
    preferences: Dict[str, Any]
    batch_size: int = 3
    min_conformity: float = 85.0
    session_id: Optional[str] = None
    new_search: bool = False


class MoreResultsIn(BaseModel):
    session_id: str
    n: int = 3


# -----------------------------
# Data
# -----------------------------
def _strip_internal(d: dict) -> dict:
    d = dict(d)
    d.pop("_centroid", None)
    d.pop("_path", None)
    # Convert _start_point tuple to flat fields RouteIn can accept
    sp = d.pop("_start_point", None)
    if sp is not None:
        d["start_lat"] = sp[0]
        d["start_lng"] = sp[1]
    return d


_RAW_GPX_ROUTES = load_routes_from_gpx_dir("./data/gpx")
ROUTE_DB = [RouteIn(**_strip_internal(r)) for r in _RAW_GPX_ROUTES]
print(f"[GPX] Loaded {len(ROUTE_DB)} routes")


# -----------------------------
# Rules-based translator
# -----------------------------
def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    p = max(0.0, min(100.0, float(p)))
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _extract_first_number(text: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


_NUM_WORDS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}
_NUM_WORD_RE = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
_DIST_UNIT = r"(?:miles?|mi|km|kilometers?|kilometres?|k)"


def _number_words_to_digits(q: str) -> str:
    def tens_units(m):
        return str(_NUM_WORDS[m.group(1)] + _NUM_WORDS[m.group(2)])

    q = re.sub(r"\b(twenty|thirty|forty)[ -](one|two|three|four|five|six|seven|eight|nine)\b", tens_units, q)
    q = re.sub(r"\b(" + _NUM_WORD_RE + r")(?=\s+(?:and|to)\s+(?:" + _NUM_WORD_RE + r"|\d)[a-z0-9. -]*?\s*(?:miles?|mi|km|kilometers?|kilometres?)\b)",
               lambda m: str(_NUM_WORDS[m.group(1)]), q)
    q = re.sub(r"\b(" + _NUM_WORD_RE + r") and a half\b", lambda m: str(_NUM_WORDS[m.group(1)] + 0.5), q)
    q = re.sub(r"\b(?:a )?half[ -](?:a[ -])?(?=(?:mile|mi|km|kilometer|kilometre)s?\b)", "0.5 ", q)
    q = re.sub(r"\b(?:a )?quarter[ -](?:of a[ -])?(?=(?:mile|mi|km|kilometer|kilometre)s?\b)", "0.25 ", q)
    q = re.sub(r"\b(" + _NUM_WORD_RE + r")(?=[ -](?:miles?|mi|km|kilometers?|kilometres?|k|ft|feet|foot|meters?|metres?|m)\b)",
               lambda m: str(_NUM_WORDS[m.group(1)]), q)
    q = re.sub(r"\b(" + _NUM_WORD_RE + r")(?=\s+(?:hundred|thousand)\b)", lambda m: str(_NUM_WORDS[m.group(1)]), q)
    q = re.sub(r"(\d+(?:\.\d+)?)\s+hundred\b", lambda m: str(float(m.group(1)) * 100), q)
    q = re.sub(r"(\d+(?:\.\d+)?)\s+thousand\b", lambda m: str(float(m.group(1)) * 1000), q)
    return q


_NUM = r"(\d+(?:,\d{3})*(?:\.\d+)?)"
_ELEV_UNIT = r"(ft|feet|foot|'|m|meters?|metres?)"
_ELEV_WORD = r"(?:of\s+)?(?:elevation(?:\s+gain)?|gain|climb(?:ing)?|ascent|vert(?:ical)?)"


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _to_ft(val: float, unit: str) -> float:
    return val if unit in {"ft", "feet", "foot", "'"} else val * 3.28084


def _extract_elevation_limits(q: str) -> Tuple[Optional[float], Optional[float], str]:
    max_ft: Optional[float] = None
    min_ft: Optional[float] = None

    def need_word(unit: str) -> bool:
        return unit not in {"ft", "feet", "foot", "'"}

    pat_max = re.compile(
        r"(?:under|less than|below|fewer than|max(?:imum)?(?: of)?|no more than|at most|up to|not more than|capped at)\s*"
        + _NUM + r"\s*" + _ELEV_UNIT + r"(?:\s+" + _ELEV_WORD + r")?(?![a-z])"
    )
    pat_min = re.compile(
        r"(?:at least|more than|over|above|min(?:imum)?(?: of)?|no less than|not less than|greater than)\s*"
        + _NUM + r"\s*" + _ELEV_UNIT + r"(?:\s+" + _ELEV_WORD + r")?(?![a-z])"
    )
    for pat, kind in ((pat_max, "max"), (pat_min, "min")):
        def repl(m, kind=kind):
            nonlocal max_ft, min_ft
            unit = m.group(2)
            has_word = bool(re.search(r"elevation|gain|climb|ascent|vert", m.group(0)))
            if need_word(unit) and not has_word:
                return m.group(0)
            ft = round(_to_ft(_to_float(m.group(1)), unit), 0)
            if kind == "max":
                max_ft = ft
            else:
                min_ft = ft
            return " "
        q = pat.sub(repl, q)
    m = re.search(r"\b" + _NUM + r"\s*" + _ELEV_UNIT + r"\s+(?:of\s+)?(?:elevation(?:\s+gain)?|gain|climb(?:ing)?)\s+(?:or less|max(?:imum)?)\b", q)
    if m and max_ft is None:
        max_ft = round(_to_ft(_to_float(m.group(1)), m.group(2)), 0)
        q = q.replace(m.group(0), " ")
    m = re.search(r"\b" + _NUM + r"\s*" + _ELEV_UNIT + r"\s+(?:of\s+)?(?:elevation(?:\s+gain)?|gain|climb(?:ing)?)\s+(?:or more|min(?:imum)?|plus)\b", q)
    if m and min_ft is None:
        min_ft = round(_to_ft(_to_float(m.group(1)), m.group(2)), 0)
        q = q.replace(m.group(0), " ")
    return max_ft, min_ft, q


_PROX_ANCHOR = r"(?:me|here|us|home|my (?:location|house|place|home)|where i am|my area|the city)"


def _extract_proximity(q: str) -> Tuple[Optional[float], str]:
    pat = re.compile(
        r"(?:within|inside|under|less than|up to|no more than|at most|max(?:imum)?)?\s*" + _NUM + r"\s*" + _DIST_UNIT
        + r"(?:\s+(?:away|drive))?\s+(?:of|from)\s+" + _PROX_ANCHOR + r"\b"
    )
    m = pat.search(q)
    pat2 = re.compile(r"\b" + _NUM + r"\s*" + _DIST_UNIT + r"\s+away\b")
    if not m:
        m = pat2.search(q)
    if not m:
        return None, q
    val = _to_float(m.group(1))
    if re.search(r"\b(?:km|kilometers?|kilometres?|k)\b", m.group(0)):
        val *= 0.621371
    return round(val, 2), q.replace(m.group(0), " ", 1)


def _contains_any(text: str, words: List[str]) -> bool:
    t = _norm_text(text)
    return any(w in t for w in words)


_MILES = [float(r.distance_miles) for r in ROUTE_DB]
_ELEV  = [float(r.elevation_gain) for r in ROUTE_DB]

MILES_P25 = _percentile(_MILES, 25)
MILES_P75 = _percentile(_MILES, 75)
MILES_P90 = _percentile(_MILES, 90)
MILES_MAX = max(_MILES) if _MILES else 0.0

DISTANCE_LIMITS: Dict[str, float] = {
    "min_target_miles": 0.5,
    "cap_over_library_max": 1.1,
}
MILES_CAP = round(max(MILES_P90, MILES_MAX * DISTANCE_LIMITS["cap_over_library_max"]), 2)
ELEV_P25  = _percentile(_ELEV, 25)
ELEV_P75  = _percentile(_ELEV, 75)


PARK_ALIASES: Dict[str, str] = {
    "almaden quicksilver": "Almaden Quicksilver County Park",
    "quicksilver": "Almaden Quicksilver County Park",
    "almaden": "Almaden Quicksilver County Park",
    "joseph d grant": "Joseph D. Grant County Park",
    "joseph d. grant": "Joseph D. Grant County Park",
    "joseph grant": "Joseph D. Grant County Park",
    "jdg": "Joseph D. Grant County Park",
    "grant park": "Joseph D. Grant County Park",
    "grant ranch": "Joseph D. Grant County Park",
    "mount madonna": "Mount Madonna County Park",
    "mt madonna": "Mount Madonna County Park",
    "mt. madonna": "Mount Madonna County Park",
    "madonna": "Mount Madonna County Park",
    "uvas canyon": "Uvas Canyon County Park",
    "uvas": "Uvas Canyon County Park",
    "santa teresa": "Santa Teresa County Park",
    "sanborn": "Sanborn County Park",
    "calero": "Calero County Park",
    "coyote lake": "Coyote Lake - Harvey Bear Ranch County Park",
    "harvey bear": "Coyote Lake - Harvey Bear Ranch County Park",
    "coyote lake harvey bear": "Coyote Lake - Harvey Bear Ranch County Park",
    "ed r levin": "Ed R. Levin County Park",
    "ed r. levin": "Ed R. Levin County Park",
    "ed levin": "Ed R. Levin County Park",
    "levin": "Ed R. Levin County Park",
    "hellyer": "Hellyer County Park",
    "lexington reservoir": "Lexington Reservoir County Park",
    "lexington": "Lexington Reservoir County Park",
    "los gatos creek": "Los Gatos Creek County Park",
    "lgct": "Los Gatos Creek County Park",
    "martial cottle": "Martial Cottle Park",
    "stevens creek": "Stevens Creek County Park",
    "upper stevens creek": "Upper Stevens Creek County Park",
    "vasona": "Vasona Lake County Park",
    "villa montalvo": "Villa Montalvo County Park",
    "montalvo": "Villa Montalvo County Park",
    "sunnyvale baylands": "Sunnyvale Baylands Park",
    "baylands": "Sunnyvale Baylands Park",
    "alviso marina": "Alviso Marina County Park",
    "alviso": "Alviso Marina County Park",
    "anderson lake": "Anderson Lake County Park",
    "anderson": "Anderson Lake County Park",
    "almaden lake": "Almaden Lake Park",
    "sierra azul": "Sierra Azul Open Space Preserve",
    "sierra azul open space preserve": "Sierra Azul Open Space Preserve",
    "castle rock": "Castle Rock State Park",
    "castle rock state park": "Castle Rock State Park",
}


def _extract_park_alias(query: str) -> Optional[str]:
    q = _norm_text(query)
    for alias in sorted(PARK_ALIASES.keys(), key=len, reverse=True):
        pattern = r"(?:^|[^a-z])" + re.escape(alias) + r"(?:[^a-z]|$)"
        if re.search(pattern, q):
            return alias
    return _fuzzy_park_alias(q)


def _fuzzy_park_alias(q: str) -> Optional[str]:
    words = re.findall(r"[a-z]+", q)
    best: Optional[Tuple[float, str]] = None
    for alias in PARK_ALIASES:
        if len(alias) < 7 or "." in alias:
            continue
        n = len(alias.split())
        for i in range(0, len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            if gram[:1] != alias[:1] or abs(len(gram) - len(alias)) > 2:
                continue
            ratio = difflib.SequenceMatcher(None, gram, alias).ratio()
            if ratio >= 0.86 and (best is None or ratio > best[0]):
                best = (ratio, alias)
    return best[1] if best else None


def _extract_park_filter(query: str) -> Optional[str]:
    alias = _extract_park_alias(query)
    return PARK_ALIASES[alias] if alias else None


def translate_query_rules(query: str, base_prefs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q = _norm_text(query)
    prefs = dict(base_prefs or {})

    if prefs.get("park_filter") is None:
        park = _extract_park_filter(query)
        if park:
            prefs["park_filter"] = park

    # Park names are not preference keywords ("Castle Rock" is not rocky, "Quicksilver" is not quick)
    park_alias = _extract_park_alias(query)
    if park_alias:
        q_exact = re.sub(r"(?:^|(?<=[^a-z]))" + re.escape(park_alias) + r"(?=[^a-z]|$)", " ", q)
        if q_exact == q:
            n = len(park_alias.split())
            ws = q.split()
            for i in range(0, len(ws) - n + 1):
                gram = " ".join(re.sub(r"[^a-z]", "", w) for w in ws[i:i + n])
                if difflib.SequenceMatcher(None, gram, park_alias).ratio() >= 0.86:
                    ws[i:i + n] = [" "] * n
                    break
            q_exact = " ".join(ws)
        q = q_exact

    q = _number_words_to_digits(q)

    max_gain_ft, min_gain_ft, q = _extract_elevation_limits(q)
    if max_gain_ft is not None and prefs.get("max_gain_ft") is None:
        prefs["max_gain_ft"] = max_gain_ft
    if min_gain_ft is not None and prefs.get("min_gain_ft") is None:
        prefs["min_gain_ft"] = min_gain_ft

    prox_mi, q = _extract_proximity(q)
    if prox_mi is not None and prefs.get("max_proximity") is None:
        prefs["max_proximity"] = prox_mi

    miles_from_text: Optional[float] = None
    range_m = re.search(
        r"(?:between\s+)?" + _NUM + r"\s*(?:" + _DIST_UNIT + r")?\s*(?:and|to|-)\s*" + _NUM + r"\s*(" + _DIST_UNIT + r")\b", q
    )
    if range_m and ("between" in range_m.group(0) or "-" in range_m.group(0) or " to " in range_m.group(0)):
        lo, hi = _to_float(range_m.group(1)), _to_float(range_m.group(2))
        if re.fullmatch(r"km|kilometers?|kilometres?|k", range_m.group(3)):
            lo, hi = lo * 0.621371, hi * 0.621371
        if lo > hi:
            lo, hi = hi, lo
        if prefs.get("min_mileage") is None and prefs.get("max_mileage") is None and prefs.get("target_miles") is None:
            prefs["min_mileage"] = round(lo, 2)
            prefs["max_mileage"] = round(hi, 2)
        q = q.replace(range_m.group(0), " ", 1)
    elif re.search(r"\b(?:under|less than|below|at most|no more than|max(?:imum)?(?: of)?|up to|shorter than)\s*" + _NUM + r"\s*(" + _DIST_UNIT + r")(?![a-z])", q):
        bm = re.search(r"\b(?:under|less than|below|at most|no more than|max(?:imum)?(?: of)?|up to|shorter than)\s*" + _NUM + r"\s*(" + _DIST_UNIT + r")(?![a-z])", q)
        hi = _to_float(bm.group(1)) * (0.621371 if re.fullmatch(r"km|kilometers?|kilometres?|k", bm.group(2)) else 1.0)
        if prefs.get("min_mileage") is None and prefs.get("max_mileage") is None and prefs.get("target_miles") is None:
            prefs["min_mileage"] = 0.0
            prefs["max_mileage"] = round(hi, 2)
        q = q.replace(bm.group(0), " ", 1)
    elif re.search(r"\b(?:at least|more than|over|no less than|minimum(?: of)?|min|longer than)\s*" + _NUM + r"\s*(" + _DIST_UNIT + r")(?![a-z])", q):
        bm = re.search(r"\b(?:at least|more than|over|no less than|minimum(?: of)?|min|longer than)\s*" + _NUM + r"\s*(" + _DIST_UNIT + r")(?![a-z])", q)
        lo = _to_float(bm.group(1)) * (0.621371 if re.fullmatch(r"km|kilometers?|kilometres?|k", bm.group(2)) else 1.0)
        if prefs.get("min_mileage") is None and prefs.get("max_mileage") is None and prefs.get("target_miles") is None:
            prefs["min_mileage"] = round(lo, 2)
            prefs["max_mileage"] = round(max(lo + 0.1, float(MILES_CAP)), 2)
        q = q.replace(bm.group(0), " ", 1)
    elif "half marathon" in q or "half-marathon" in q or "halfmarathon" in q:
        miles_from_text = 13.1094
    elif "marathon" in q:
        miles_from_text = 26.2188
    else:
        dm = re.search(r"(?<![\w.])" + _NUM + r"\s*(" + _DIST_UNIT + r")(?![a-z])", q)
        if dm:
            n = _to_float(dm.group(1))
            unit = dm.group(2)
            miles_from_text = n * 0.621371 if re.fullmatch(r"km|kilometers?|kilometres?|k", unit) else n
        else:
            bare = re.sub(r"\b\d+(?:\.\d+)?\s*(?:min(?:ute)?s?|hrs?|hours?|ft|feet|foot|m|meters?|metres?|%)\b", " ", q)
            n = _extract_first_number(bare)
            if n is not None:
                miles_from_text = n

    if miles_from_text is not None and prefs.get("target_miles") is None and (
        prefs.get("min_mileage") is None and prefs.get("max_mileage") is None
    ):
        prefs["target_miles"] = round(float(miles_from_text), 2)

    if prefs.get("target_miles") is None and prefs.get("min_mileage") is None and prefs.get("max_mileage") is None:
        if _contains_any(q, ["short", "quick", "brief", "not long"]):
            prefs["min_mileage"] = 0.0
            prefs["max_mileage"] = round(max(1.0, MILES_P25), 2)
        elif _contains_any(q, ["long", "far", "epic", "endurance"]):
            prefs["min_mileage"] = round(max(1.0, MILES_P75), 2)
            prefs["max_mileage"] = round(max(prefs["min_mileage"] + 0.1, MILES_P90), 2)

    # Elevation / steepness language
    flat_words = [
        "flat", "no hills", "no hill", "not hilly", "avoid hills", "no climbing", "no climbs", "no elevation",
        "minimal elevation", "minimal climb", "minimal gain", "little elevation", "little climbing",
        "low climb", "low elevation", "gentle", "pancake", "level ground",
    ]
    if prefs.get("max_elevation") is None and prefs.get("target_elevation_gain") is None:
        if _contains_any(q, flat_words):
            prefs["max_elevation"] = round(max(100.0, ELEV_P25), 0)
            prefs["flat"] = True
        elif _contains_any(q, ["not steep", "not too steep", "easy"]):
            prefs["max_elevation"] = round(max(100.0, ELEV_P25), 0)
        elif prefs.get("min_gain_ft") is not None:
            pass
        elif _contains_any(q, ["steep", "hilly", "climb", "vert", "mountain"]):
            prefs["target_elevation_gain"] = round(max(300.0, ELEV_P75), 0)
    elif prefs.get("flat") is None and _contains_any(q, flat_words):
        prefs["flat"] = True

    # Shade / Views / Crowds
    if prefs.get("shade_preference") is None:
        if _contains_any(q, ["shady", "shade", "cool", "tree", "canopy"]):
            prefs["shade_preference"] = 0.8
        elif _contains_any(q, ["sunny", "no shade", "open", "exposed"]):
            prefs["shade_preference"] = 0.2

    if prefs.get("views_preference") is None:
        if _contains_any(q, ["views", "scenic", "pretty", "ridgeline", "lookout", "vista", "panorama"]):
            prefs["views_preference"] = 0.8

    if prefs.get("crowds_preference") is None:
        if _contains_any(q, ["not crowded", "avoid crowds", "quiet", "secluded", "empty", "uncrowded"]):
            prefs["crowds_preference"] = "secluded"
        elif _contains_any(q, ["popular", "busy", "crowded", "lots of people"]):
            prefs["crowds_preference"] = "popular"

    # Proximity language
    if prefs.get("max_proximity") is None:
        if _contains_any(q, ["close", "near", "nearby", "around here", "local"]):
            prefs["max_proximity"] = 10.0
        elif _contains_any(q, ["far", "worth the drive", "drive", "day trip"]):
            prefs["max_proximity"] = 35.0

    # Route shape intent
    if prefs.get("intent") is None:
        if _contains_any(q, ["loop", "loop trail", "circular"]):
            prefs["intent"] = "loop"
        elif _contains_any(q, ["out and back", "out-and-back", "there and back"]):
            prefs["intent"] = "out-and-back"

    # Difficulty preference (NEW - 1b)
    if prefs.get("difficulty_preference") is None:
        if _contains_any(q, ["easy", "beginner", "gentle", "simple", "relaxed", "chill", "mellow"]):
            prefs["difficulty_preference"] = "easy"
        elif _contains_any(q, ["moderate", "intermediate", "medium"]):
            prefs["difficulty_preference"] = "moderate"
        elif _contains_any(q, ["hard", "difficult", "challenging", "strenuous", "tough"]):
            prefs["difficulty_preference"] = "hard"
        elif _contains_any(q, ["extreme", "expert", "very hard", "brutal", "grueling"]):
            prefs["difficulty_preference"] = "very hard"

    # ========================================================
    # OSM-DRIVEN PREFERENCES (Stage C)
    # ========================================================

    # Surface preference (paved / dirt / gravel / rocky)
    if prefs.get("surface_pref") is None:
        if _contains_any(q, ["paved", "asphalt", "concrete", "road bike", "stroller", "smooth surface"]):
            prefs["surface_pref"] = "paved"
        elif _contains_any(q, ["dirt", "trail", "singletrack", "single track", "off road", "off-road"]):
            prefs["surface_pref"] = "dirt"
        elif _contains_any(q, ["gravel", "fire road", "fireroad", "doubletrack", "double track"]):
            prefs["surface_pref"] = "gravel"
        elif _contains_any(q, ["rocky", "rock", "boulder", "scrambly", "scramble"]):
            prefs["surface_pref"] = "rocky"

    # Facilities (parking, restrooms, water)
    if prefs.get("wants_facilities") is None:
        if _contains_any(q, [
            "parking", "with parking", "trailhead parking", "park nearby",
            "restroom", "bathroom", "toilet", "facilities", "amenities",
            "water fountain", "drinking water", "refill", "water available",
            "with amenities", "well-equipped",
        ]):
            prefs["wants_facilities"] = True

    # Scenic POIs (peaks, viewpoints, waterfalls)
    if prefs.get("views_preference") is None or prefs.get("views_preference") == 0.8:
        if _contains_any(q, [
            "waterfall", "waterfalls", "peak", "summit", "viewpoint",
            "overlook", "lookout", "scenic point", "vista point",
        ]):
            prefs["views_preference"] = 0.85

    # Dog-friendly (soft preference + hard requirement)
    if prefs.get("has_dog") is None:
        if _contains_any(q, [
            "dog", "dogs", "with my dog", "with dog", "puppy", "pup",
            "dog friendly", "dog-friendly", "bring my dog",
        ]):
            prefs["has_dog"] = True
            # If user explicitly says "must allow dogs" or similar → hard gate
            if _contains_any(q, [
                "must allow dogs", "dogs required", "dogs must", "dogs allowed only",
                "where dogs are allowed",
            ]):
                prefs["require_dog_allowed"] = True

    # Bike-legal (soft preference becomes hard gate when intent is clear)
    if prefs.get("require_bike_legal") is None and not re.search(r"\b(?:no|without|not|avoid|except)\s+(?:any\s+)?(?:bikes?|bikers?|cyclists?|cycling|biking)\b", q):
        if _contains_any(q, [
            "bike", "biking", "cycling", "cyclist", "mtb", "mountain bike",
            "mountain biking", "ride", "riding", "gravel bike", "road bike",
        ]):
            prefs["require_bike_legal"] = True

    # Wheelchair / stroller / accessibility (always hard gate)
    if prefs.get("require_wheelchair_accessible") is None:
        if _contains_any(q, [
            "wheelchair", "accessible", "ada", "stroller", "stroller-friendly",
            "wheelchair accessible", "paved only", "paved path only",
        ]):
            prefs["require_wheelchair_accessible"] = True

    # Technicality preference
    if prefs.get("technicality_pref") is None:
        if _contains_any(q, ["non-technical", "non technical", "smooth", "easy footing", "beginner friendly"]):
            prefs["technicality_pref"] = "low"
        elif _contains_any(q, ["technical", "rooty", "rocky terrain", "rugged"]):
            prefs["technicality_pref"] = "high"
        elif _contains_any(q, ["moderate technical", "intermediate technical"]):
            prefs["technicality_pref"] = "medium"

    return prefs


def _prefs_have_enough_signal(prefs: Dict[str, Any]) -> bool:
    keys = [
        "target_miles", "min_mileage", "max_mileage",
        "target_elevation_gain", "max_elevation",
        "shade_preference", "views_preference", "crowds_preference",
        "max_proximity", "preferred_surface", "allowed_surface_types",
        "location", "difficulty_preference",
        # OSM-driven preferences
        "surface_pref", "wants_facilities", "has_dog",
        "require_bike_legal", "require_dog_allowed",
        "require_wheelchair_accessible", "technicality_pref",
    ]
    count = 0
    for k in keys:
        if prefs.get(k) is not None:
            v = prefs.get(k)
            if isinstance(v, str) and not v.strip():
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            count += 1
    return count >= 2


def _validate_and_clamp_prefs(p: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(p, dict):
        return out

    def num(v) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    tm = num(p.get("target_miles"))
    if tm is not None and tm > 0:
        out["target_miles"] = round(tm, 2)

    mn = num(p.get("min_mileage"))
    mx = num(p.get("max_mileage"))
    if mn is not None or mx is not None:
        mn = 0.0 if mn is None else max(0.0, mn)
        mx = 100.0 if mx is None else max(0.0, mx)
        if mx < mn:
            mn, mx = mx, mn
        out["min_mileage"] = round(mn, 2)
        out["max_mileage"] = round(mx, 2)

    te = num(p.get("target_elevation_gain"))
    if te is not None and te >= 0:
        out["target_elevation_gain"] = round(te, 0)

    me = num(p.get("max_elevation"))
    if me is not None and me >= 0:
        out["max_elevation"] = round(me, 0)

    sp = num(p.get("shade_preference"))
    if sp is not None:
        out["shade_preference"] = round(_clamp(sp, 0.0, 1.0), 2)

    vp = num(p.get("views_preference"))
    if vp is not None:
        out["views_preference"] = round(_clamp(vp, 0.0, 1.0), 2)

    cp = p.get("crowds_preference")
    if isinstance(cp, str):
        cpn = cp.strip().lower()
        if cpn in {"popular", "secluded", "balanced"}:
            out["crowds_preference"] = cpn

    mp = num(p.get("max_proximity"))
    if mp is not None and mp > 0:
        out["max_proximity"] = round(_clamp(mp, 0.1, 200.0), 2)

    for gk in ("max_gain_ft", "min_gain_ft"):
        gv = num(p.get(gk))
        if gv is not None and gv >= 0:
            out[gk] = round(min(gv, 30000.0), 0)

    if isinstance(p.get("flat"), bool):
        out["flat"] = p["flat"]

    ps = p.get("preferred_surface")
    if isinstance(ps, str):
        psn = ps.strip().lower()
        if psn in {"dirt", "paved", "mixed"}:
            out["preferred_surface"] = psn

    loc = p.get("location")
    if isinstance(loc, str) and loc.strip():
        out["location"] = loc.strip()

    pf = p.get("park_filter")
    if isinstance(pf, str) and pf.strip():
        out["park_filter"] = pf.strip()

   # Difficulty preference validation (NEW - 1b)
    dp = p.get("difficulty_preference")
    if isinstance(dp, str):
        dpn = dp.strip().lower()
        if dpn in {"easy", "moderate", "hard", "very hard"}:
            out["difficulty_preference"] = dpn

    # Intent validation
    it = p.get("intent")
    if isinstance(it, str):
        itn = it.strip().lower()
        if itn in {"loop", "out-and-back"}:
            out["intent"] = itn

    # OSM-driven preferences
    spref = p.get("surface_pref")
    if isinstance(spref, str):
        spn = spref.strip().lower()
        if spn in {"paved", "dirt", "gravel", "rocky", "any"}:
            out["surface_pref"] = spn

    tpref = p.get("technicality_pref")
    if isinstance(tpref, str):
        tpn = tpref.strip().lower()
        if tpn in {"low", "medium", "high"}:
            out["technicality_pref"] = tpn

    # Boolean OSM flags
    for bool_field in ("wants_facilities", "has_dog",
                       "require_bike_legal", "require_dog_allowed",
                       "require_wheelchair_accessible"):
        bv = p.get(bool_field)
        if isinstance(bv, bool):
            out[bool_field] = bv

    w = p.get("weights")
    if isinstance(w, dict):
        allowed = {"mileage", "elevation", "views", "proximity", "shade", "crowds", "difficulty",
            # OSM-driven dimensions
            "surface", "facilities", "scenic_pois", "dog_friendly", "technicality",
        }
        w_out = {}
        for k, v in w.items():
            if k in allowed:
                fv = num(v)
                if fv is not None and fv >= 0:
                    w_out[k] = float(fv)
        if w_out:
            out["weights"] = w_out

    return out


# -----------------------------
# Guardrails ("switches") applied AFTER LLM output
# -----------------------------
def _parse_minutes_from_query(query: str) -> Optional[float]:
    q = _norm_text(query)
    m1 = re.search(r"\b(\d+(?:\.\d+)?)\s*(min|mins|minute|minutes)\b", q)
    if m1:
        try:
            return float(m1.group(1))
        except Exception:
            return None
    m2 = re.search(r"\b(\d+(?:\.\d+)?)\s*(hr|hrs|hour|hours)\b", q)
    if m2:
        try:
            return float(m2.group(1)) * 60.0
        except Exception:
            return None
    m3 = re.search(r"\b(\d+(?:\.\d+)?)\s*-\s*minute\b", q)
    if m3:
        try:
            return float(m3.group(1))
        except Exception:
            return None
    return None


def _infer_pace_min_per_mile(query: str) -> float:
    q = _norm_text(query)
    if any(w in q for w in ["easy", "recovery", "chill", "relaxed", "jog"]):
        return 10.0
    if any(w in q for w in ["tempo", "fast", "hard", "workout", "threshold"]):
        return 8.0
    return 9.0


def apply_llm_guardrails(query: str, prefs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(prefs or {})
    q = _norm_text(query)

    if any(phrase in q for phrase in ["not crowded", "avoid crowds", "no crowds", "uncrowded", "quiet", "secluded"]):
        out["crowds_preference"] = "secluded"
    elif any(phrase in q for phrase in ["crowded", "busy", "popular"]):
        if any(phrase in q for phrase in ["want crowds", "popular spot", "busy vibe", "lively"]):
            out["crowds_preference"] = "popular"

    minutes = _parse_minutes_from_query(query)
    if minutes is not None:
        pace = _infer_pace_min_per_mile(query)
        miles = minutes / max(1.0, pace)
        miles = _clamp(miles, 1.0, 15.0)
        out["target_miles"] = round(miles, 2)
        if out.get("min_mileage") is not None or out.get("max_mileage") is not None:
            out.pop("min_mileage", None)
            out.pop("max_mileage", None)

    if out.get("target_miles") is not None:
        try:
            requested_miles = float(out["target_miles"])
            out["target_miles"] = round(_clamp(requested_miles, DISTANCE_LIMITS["min_target_miles"], float(MILES_CAP)), 2)
            if abs(out["target_miles"] - requested_miles) > 0.01:
                out["requested_miles"] = round(requested_miles, 2)
        except Exception:
            out.pop("target_miles", None)

    if out.get("min_mileage") is not None or out.get("max_mileage") is not None:
        try:
            mn = float(out.get("min_mileage", 0.0) or 0.0)
            mx = float(out.get("max_mileage", float(MILES_CAP)) or float(MILES_CAP))
            mn = _clamp(mn, 0.0, float(MILES_CAP))
            mx = _clamp(mx, 0.0, float(MILES_CAP))
            if mx < mn:
                mn, mx = mx, mn
            out["min_mileage"] = round(mn, 2)
            out["max_mileage"] = round(mx, 2)
        except Exception:
            out.pop("min_mileage", None)
            out.pop("max_mileage", None)

    if "easy" in q and out.get("max_elevation") is None and out.get("target_elevation_gain") is None:
        out["max_elevation"] = round(max(100.0, float(ELEV_P25)), 0)

    return out


def merge_prefs(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (incoming or {}).items():
        if v is None:
            continue
        if k in out and out[k] is not None and not (isinstance(out[k], str) and not out[k].strip()):
            continue
        out[k] = v
    return out


def translate_query_llm(query: str, base_prefs: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM fallback translation using Structured Outputs.
    Returns dict: {"prefs": {...}, "confidence": float}
    """
    q = query.strip()
    cache_key = f"{OPENAI_MODEL}::{_norm_text(q)}::{json.dumps(base_prefs, sort_keys=True)}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    prefs_required = [
        "target_miles", "min_mileage", "max_mileage",
        "target_elevation_gain", "max_elevation",
        "shade_preference", "views_preference", "crowds_preference",
        "max_proximity", "preferred_surface", "location", "weights",
        "intent", "difficulty_preference", "flat", "max_gain_ft", "min_gain_ft",
        # OSM-driven preferences
        "surface_pref", "wants_facilities", "has_dog",
        "require_bike_legal", "require_dog_allowed",
        "require_wheelchair_accessible", "technicality_pref",
    ]
    weights_required = ["mileage", "elevation", "views", "proximity", "shade", "crowds"]

    schema = {
        "name": "route_prefs",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "confidence": {"type": "number"},
                "prefs": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_miles":          {"type": ["number", "null"]},
                        "min_mileage":           {"type": ["number", "null"]},
                        "max_mileage":           {"type": ["number", "null"]},
                        "target_elevation_gain": {"type": ["number", "null"]},
                        "max_elevation":         {"type": ["number", "null"]},
                        "shade_preference":      {"type": ["number", "null"]},
                        "views_preference":      {"type": ["number", "null"]},
                        "crowds_preference": {
                            "anyOf": [
                                {"type": "string", "enum": ["popular", "secluded", "balanced"]},
                                {"type": "null"},
                            ]
                        },
                        "max_proximity":    {"type": ["number", "null"]},
                        "flat":             {"type": ["boolean", "null"]},
                        "max_gain_ft":      {"type": ["number", "null"]},
                        "min_gain_ft":      {"type": ["number", "null"]},
                        "preferred_surface": {
                            "anyOf": [
                                {"type": "string", "enum": ["dirt", "paved", "mixed"]},
                                {"type": "null"},
                            ]
                        },
                        "intent": {
                            "anyOf": [
                                {"type": "string", "enum": ["loop", "out-and-back"]},
                                {"type": "null"},
                            ]
                        },
                        "difficulty_preference": {
                            "anyOf": [
                                {"type": "string", "enum": ["easy", "moderate", "hard", "very hard"]},
                                {"type": "null"},
                            ]
                        },
                        "surface_pref": {
                            "anyOf": [
                                {"type": "string", "enum": ["paved", "dirt", "gravel", "rocky", "any"]},
                                {"type": "null"},
                            ]
                        },
                        "wants_facilities":              {"type": ["boolean", "null"]},
                        "has_dog":                       {"type": ["boolean", "null"]},
                        "require_bike_legal":            {"type": ["boolean", "null"]},
                        "require_dog_allowed":           {"type": ["boolean", "null"]},
                        "require_wheelchair_accessible": {"type": ["boolean", "null"]},
                        "technicality_pref": {
                            "anyOf": [
                                {"type": "string", "enum": ["low", "medium", "high"]},
                                {"type": "null"},
                            ]
                        },
                        "location": {"type": ["string", "null"]},
                        "weights": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "mileage":   {"type": ["number", "null"]},
                                        "elevation": {"type": ["number", "null"]},
                                        "views":     {"type": ["number", "null"]},
                                        "proximity": {"type": ["number", "null"]},
                                        "shade":     {"type": ["number", "null"]},
                                        "crowds":    {"type": ["number", "null"]},
                                    },
                                    "required": weights_required,
                                },
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": prefs_required,
                },
                "notes": {"type": ["string", "null"]},
            },
            "required": ["confidence", "prefs", "notes"],
        },
        "strict": True,
    }

    system = (
        "You translate a user's route request into engine preferences JSON.\n"
        "Rules:\n"
        "1) If user mentions TIME (minutes/hours), DO NOT treat it as miles. Convert time -> miles using pace:\n"
        "   - easy/recovery/jog: ~10:00 min/mile\n"
        "   - default: ~9:00 min/mile\n"
        "   - fast/tempo/hard: ~8:00 min/mile\n"
        "2) Handle negation: 'not crowded'/'avoid crowds' => crowds_preference='secluded'.\n"
        "3) Set target_miles to the user's explicit distance without clamping it; the server enforces the distance cap.\n"
        "4) Only use keys in the schema. If unknown, set null.\n"
        "Output MUST match the JSON schema strictly."
    )

    user = {
        "query": q,
        "existing_preferences": base_prefs,
        "dataset_context": {
            "miles_p25": MILES_P25,
            "miles_p75": MILES_P75,
            "miles_p90": MILES_P90,
            "miles_max": MILES_MAX,
            "miles_cap": MILES_CAP,
            "elev_p25":  ELEV_P25,
            "elev_p75":  ELEV_P75,
        },
    }

    client = _openai_client()
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user",   "content": json.dumps(user, ensure_ascii=False)},
            ],
            text={"format": {"type": "json_schema", "json_schema": schema}},
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
    except TypeError:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user",   "content": json.dumps(user, ensure_ascii=False)},
            ],
            text={"format": {"type": "json_schema", "json_schema": schema}},
        )

    raw = (resp.output_text or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"confidence": 0.0, "prefs": {}, "notes": None}

    conf = 0.0
    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0

    prefs = _validate_and_clamp_prefs(parsed.get("prefs", {}))
    payload = {"confidence": conf, "prefs": prefs}
    _cache_set(cache_key, payload)
    return payload


# -----------------------------
# Score bands + batching logic
# -----------------------------
BAND_THRESHOLDS: List[float] = [85.0, 55.0, 40.0]
MIN_RECOMMEND_SCORE = 40.0


def _bands_from_start_threshold(start_threshold: float) -> List[float]:
    start = 40.0
    for t in BAND_THRESHOLDS:
        if start_threshold >= t:
            start = t
            break
    return [t for t in BAND_THRESHOLDS if t <= start]


def _pull_with_threshold(
    ranked: List[Dict[str, Any]],
    shown: Set[str],
    threshold: float,
    limit: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in ranked:
        rid = item.get("route_id")
        if not rid:
            continue
        if rid in shown:
            continue
        score = float(item.get("conformity_score", 0.0))
        if score < threshold:
            break
        shown.add(rid)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _next_batch_banded(
    ranked: List[Dict[str, Any]],
    shown: Set[str],
    batch_size: int,
    start_threshold: float,
) -> Tuple[List[Dict[str, Any]], float, int]:
    thresholds = _bands_from_start_threshold(start_threshold)
    out: List[Dict[str, Any]] = []
    used_threshold = thresholds[0] if thresholds else MIN_RECOMMEND_SCORE
    relax_level = 0
    for i, thr in enumerate(thresholds):
        need = batch_size - len(out)
        if need <= 0:
            break
        pulled = _pull_with_threshold(ranked=ranked, shown=shown, threshold=thr, limit=need)
        if pulled:
            out.extend(pulled)
            used_threshold = thr
            relax_level = i
    return out, used_threshold, relax_level


def _remaining_recommendable(ranked: List[Dict[str, Any]], shown: Set[str]) -> int:
    count = 0
    for item in ranked:
        rid = item.get("route_id")
        if not rid or rid in shown:
            continue
        score = float(item.get("conformity_score", 0.0))
        if score < MIN_RECOMMEND_SCORE:
            break
        count += 1
    return count


# -----------------------------
# Progressive Relaxation (soft constraints only)
# -----------------------------
def apply_progressive_relaxation(prefs: Dict[str, Any], relax_level: int) -> Dict[str, Any]:
    p = dict(prefs or {})
    rl = max(0, int(relax_level or 0))
    p["relax_level"] = rl

    min_m = float(p.get("min_mileage", 0.0) or 0.0)
    max_m = float(p.get("max_mileage", 100.0) or 100.0)
    if max_m < min_m:
        min_m, max_m = max_m, min_m
    mid = (min_m + max_m) / 2.0
    half = max(0.1, (max_m - min_m) / 2.0)
    widen = 1.0 + min(0.40, 0.10 * rl)
    new_half = half * widen
    p["min_mileage"] = max(0.0, mid - new_half)
    p["max_mileage"] = max(mid + new_half, p["min_mileage"] + 0.1)

    if not p.get("allowed_surface_types"):
        if rl >= 3 and p.get("preferred_surface") and str(p["preferred_surface"]).strip().lower() != "mixed":
            p["preferred_surface"] = "mixed"

    return p


MAKE_INGRESS_KEY = os.getenv("MAKE_INGRESS_KEY", "")


def require_make_key(x_make_key: str | None) -> None:
    if not MAKE_INGRESS_KEY:
        raise HTTPException(status_code=500, detail="MAKE_INGRESS_KEY not set")
    if not x_make_key or x_make_key != MAKE_INGRESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-Make-Key")


class MakeIngressBody(BaseModel):
    query: str
    preferences: Preferences = Field(default_factory=Preferences)
    batch_size: int = 3
    min_conformity: int = 85
    session_id: Optional[str] = None
    new_search: bool = False


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"ok": True}

import pdf_maps
from fastapi.responses import Response
from collections import OrderedDict
import threading

_RAW_BY_ID = {r["route_id"]: r for r in _RAW_GPX_ROUTES}


def _name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


_ID_BY_NAME_KEY = {}
for _r in _RAW_GPX_ROUTES:
    _ID_BY_NAME_KEY.setdefault(_name_key(_r["name"]), _r["route_id"])


MAP_PDF_CACHE_MAX_BYTES = int(os.getenv("MAP_PDF_CACHE_MAX_BYTES", str(16 * 1024 * 1024)))
_MAP_PDF_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_MAP_PDF_CACHE_BYTES = 0
_MAP_PDF_LOCK = threading.Lock()


def _route_pdf(route_id: str):
    global _MAP_PDF_CACHE_BYTES
    with _MAP_PDF_LOCK:
        hit = _MAP_PDF_CACHE.get(route_id)
        if hit is not None:
            _MAP_PDF_CACHE.move_to_end(route_id)
            return hit
    r = _RAW_BY_ID[route_id]
    result = pdf_maps.render_route_map(r["_path"], r["name"], r["distance_miles"], r["elevation_gain"])
    size = len(result[0])
    if size <= MAP_PDF_CACHE_MAX_BYTES:
        with _MAP_PDF_LOCK:
            if route_id not in _MAP_PDF_CACHE:
                _MAP_PDF_CACHE[route_id] = result
                _MAP_PDF_CACHE_BYTES += size
            while _MAP_PDF_CACHE_BYTES > MAP_PDF_CACHE_MAX_BYTES and _MAP_PDF_CACHE:
                _, old = _MAP_PDF_CACHE.popitem(last=False)
                _MAP_PDF_CACHE_BYTES -= len(old[0])
    return result


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


@app.get("/routes/{route_id}/map.pdf")
def route_map_pdf(route_id: str, _: None = Depends(require_api_key)):
    rid = route_id if route_id in _RAW_BY_ID else _ID_BY_NAME_KEY.get(_name_key(route_id))
    if not rid:
        return JSONResponse(status_code=404, content={"detail": "Route not found"})
    try:
        pdf, mode, park = _route_pdf(rid)
    except pdf_maps.MapUnavailable as e:
        return JSONResponse(status_code=503, content={"detail": str(e)}, headers={"Retry-After": "300"})
    except Exception as e:
        print(f"[map.pdf] render failed for {rid}: {type(e).__name__}: {e}")
        return JSONResponse(status_code=500, content={"detail": "Map render failed"})
    route_slug = _slug(_RAW_BY_ID[rid]["name"])[:80].strip("-") or "route"
    filename = f"{park}-{route_slug}.pdf" if park else f"{route_slug}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, max-age=86400",
            "X-Map-Mode": mode,
        },
    )

# -----------------------------
# Core search helper (shared by /start_search and Make ingress)
# -----------------------------
KM_TO_MI = 0.621371
M_TO_FT = 3.28084


def _location_pref_to_proximity(prefs: Dict[str, Any]) -> Optional[float]:
    loc = prefs.get("location")
    if not isinstance(loc, dict):
        return None
    prefs.pop("location", None)
    if loc.get("lat") is not None and loc.get("lng") is not None:
        prefs.setdefault("lat", float(loc["lat"]))
        prefs.setdefault("lng", float(loc["lng"]))
    radius_km = loc.get("radius_km")
    return float(radius_km) * KM_TO_MI if radius_km is not None and float(radius_km) > 0 else None


def _with_exhausted_notice(
    notice: Optional[str],
    has_more: bool,
    prefs: Dict[str, Any],
    relaxed_steps: List[str],
) -> Optional[str]:
    if has_more:
        return notice
    extra = proximity_exhausted_notice(prefs, relaxed_steps)
    parts = [n for n in (notice, extra) if n]
    return " ".join(parts) if parts else None


def _start_search_core(body: StartSearchBody) -> Dict[str, Any]:
    prefs: Dict[str, Any] = body.preferences.model_dump(exclude_none=True) if body.preferences else {}
    weights = prefs.pop("weights", None)
    radius_mi = _location_pref_to_proximity(prefs)
    gain_m = prefs.get("elevation_gain_m")
    if isinstance(gain_m, dict) and gain_m.get("max") is not None:
        prefs["max_gain_ft"] = round(float(gain_m["max"]) * M_TO_FT, 0)

    query = (body.query or "").strip()

    if query:
        prefs_rules = translate_query_rules(query, base_prefs=prefs)
        prefs = prefs_rules

        if LLM_TRANSLATION_ENABLED and not _prefs_have_enough_signal(prefs_rules):
            llm = translate_query_llm(query, base_prefs=prefs_rules)
            if float(llm.get("confidence", 0.0)) >= OPENAI_MIN_CONFIDENCE:
                merged = merge_prefs(prefs_rules, llm.get("prefs", {}))
                prefs = apply_llm_guardrails(query, merged)

        prefs = apply_llm_guardrails(query, prefs)

    if radius_mi is not None:
        prefs["max_proximity"] = min(float(prefs.get("max_proximity", radius_mi)), radius_mi)

    start_new = bool(body.new_search) or not body.session_id
    if start_new:
        session_data = {
            "created_at": int(time.time()),
            "prefs": prefs,
            "weights": weights,
            "shown": [],
            "relax_level": 0,
        }
    else:
        session_data = read_session_token(body.session_id, max_age_seconds=3600)
        if prefs:
            session_data["prefs"] = prefs
            session_data["weights"] = weights

    session_data.setdefault("created_at", int(time.time()))
    session_data.setdefault("shown", [])
    session_data.setdefault("prefs", {})
    session_data.setdefault("weights", None)
    session_data.setdefault("relax_level", 0)

    shown_set: Set[str] = set(session_data.get("shown", []) or [])

    relaxed_prefs = apply_progressive_relaxation(
        session_data["prefs"],
        int(session_data.get("relax_level", 0) or 0),
    )

    ranked, notice, relaxed_steps = select_routes_with_relaxation_steps(
        [r.to_engine_route() for r in ROUTE_DB],
        relaxed_prefs,
        weights=session_data.get("weights"),
    )

    batch, used_min_conf, band_relax_level = _next_batch_banded(
        ranked=ranked,
        shown=shown_set,
        batch_size=int(body.batch_size),
        start_threshold=float(body.min_conformity),
    )

    remaining = _remaining_recommendable(ranked, shown_set)
    has_more = remaining > 0
    notice = _with_exhausted_notice(notice, has_more, relaxed_prefs, relaxed_steps)

    shown_list = list(shown_set)
    if len(shown_list) > 2000:
        shown_list = shown_list[:2000]
    session_data["shown"] = shown_list

    session_id_token = make_session_token(session_data)

    return {
        "session_id": session_id_token,
        "routes": batch,
        "has_more": has_more,
        "min_conformity": used_min_conf,
        "relax_level": band_relax_level,
        "progressive_relax_level": int(session_data.get("relax_level", 0) or 0),
        "notice": notice,
    }


# -----------------------------
# Normal app API (uses your existing API key auth)
# -----------------------------
@app.post("/start_search")
def start_search(body: StartSearchBody, _: None = Depends(require_api_key)):
    return _start_search_core(body)


@app.post("/more_results")
def more_results(req: MoreResultsIn, _: None = Depends(require_api_key)):
    session_data = read_session_token(req.session_id, max_age_seconds=3600)
    session_data.setdefault("shown", [])
    session_data.setdefault("prefs", {})
    session_data.setdefault("weights", None)
    session_data.setdefault("relax_level", 0)
    session_data["relax_level"] = min(int(session_data.get("relax_level", 0) or 0) + 1, 8)

    shown_set: Set[str] = set(session_data.get("shown", []) or [])
    relaxed_prefs = apply_progressive_relaxation(session_data["prefs"], int(session_data["relax_level"]))

    ranked, notice, relaxed_steps = select_routes_with_relaxation_steps(
        [r.to_engine_route() for r in ROUTE_DB],
        relaxed_prefs,
        weights=session_data.get("weights"),
    )

    batch, used_min_conf, band_relax_level = _next_batch_banded(
        ranked=ranked,
        shown=shown_set,
        batch_size=int(req.n),
        start_threshold=85.0,
    )

    remaining = _remaining_recommendable(ranked, shown_set)
    has_more = remaining > 0
    notice = _with_exhausted_notice(notice, has_more, relaxed_prefs, relaxed_steps)

    shown_list = list(shown_set)
    if len(shown_list) > 2000:
        shown_list = shown_list[:2000]
    session_data["shown"] = shown_list
    session_data["created_at"] = int(time.time())

    session_id_token = make_session_token(session_data)

    return {
        "session_id": session_id_token,
        "routes": batch,
        "has_more": has_more,
        "min_conformity": used_min_conf,
        "relax_level": band_relax_level,
        "progressive_relax_level": int(session_data.get("relax_level", 0) or 0),
        "notice": notice,
    }


# -----------------------------
# Make ingress (Option B: simple separate key)
# -----------------------------
MAKE_INGRESS_KEY = os.getenv("MAKE_INGRESS_KEY", "")


def require_make_key(x_make_key: str | None) -> None:
    if not MAKE_INGRESS_KEY:
        raise HTTPException(status_code=500, detail="MAKE_INGRESS_KEY not set")
    if not x_make_key or x_make_key != MAKE_INGRESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-Make-Key")


class MakeIngressBody(BaseModel):
    # What Make sends AFTER it has translated NL -> typed Preferences
    query: str
    preferences: Preferences = Field(default_factory=Preferences)
    batch_size: int = 3
    min_conformity: int = 85
    session_id: Optional[str] = None
    new_search: bool = False


@app.post("/make/translate_and_search")
def make_translate_and_search(
    payload: MakeIngressBody,
    x_make_key: str | None = Header(default=None, alias="X-Make-Key"),
):
    require_make_key(x_make_key)
    body = StartSearchBody(
        query=payload.query,
        preferences=payload.preferences,
        batch_size=payload.batch_size,
        min_conformity=payload.min_conformity,
        session_id=payload.session_id,
        new_search=payload.new_search,
    )
    return _start_search_core(body)


# -----------------------------
# Errors
# -----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if DEV_MODE:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc), "type": exc.__class__.__name__},
        )
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )

