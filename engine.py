from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from gpx_loader import haversine_miles


@dataclass
class Route:
    route_id: str
    name: str
    location: str
    distance_miles: float
    elevation_gain: float
    surface_type: str
    shade_pct: float          # 0-100 (legacy; prefer osm_shade_pct)
    scenic_likelihood: float  # 0-1 (legacy heuristic)
    proximity_miles: float
    route_type: str
    popularity: float         # 0-100 (optional signal)
    difficulty: str
    technicality: str
    # Computed at load time from GPX data (optional — default for legacy data)
    max_grade_pct: Optional[float] = 0.0
    avg_grade_pct: Optional[float] = 0.0
    _start_point: Optional[Tuple[float, float]] = None
    _centroid: Optional[Tuple[float, float]] = None
    _path: Optional[str] = None
    # ----------------------------------------------------------------
    # OSM-derived fields (Santa Clara County, OpenStreetMap ground truth)
    # All optional with safe defaults so legacy callers don't break.
    # ----------------------------------------------------------------
    osm_surface: str = "unknown"
    osm_highway: str = "unknown"
    osm_smoothness: str = ""
    osm_bicycle_legal: bool = True
    osm_horse_legal: bool = False
    osm_dog_allowed: Optional[bool] = None
    osm_technicality: float = 0.0      # 0-6 scale (mtb:scale or sac_scale derived)
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
    osm_picnic_count: int = 0
    osm_camping_count: int = 0


# Band thresholds (your spec)
BAND_1_MIN = 85.00
BAND_2_MIN = 55.00
BAND_3_MIN = 40.00   # below this is not recommended


DEFAULT_WEIGHTS = {
    "mileage":      0.22,
    "elevation":    0.15,
    "difficulty":   0.10,
    "proximity":    0.10,
    "views":        0.10,
    "shade":        0.08,
    "surface":      0.07,
    "crowds":       0.05,
    "facilities":   0.05,
    "scenic_pois":  0.05,
    "dog_friendly": 0.02,
    "technicality": 0.01,
}
# Sum = 1.00


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_text(s: Optional[str]) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _location_match(route_loc: str, user_loc: str) -> bool:
    """
    v1 location gate: forgiving string match (no geometry).
    Accept if either string contains the other after normalization.
    """
    rl = _norm_text(route_loc)
    ul = _norm_text(user_loc)
    if not ul:
        return True
    if not rl:
        return False
    return (ul in rl) or (rl in ul)


# ------------------------------------------------------------
# Scenic likelihood inference
# ------------------------------------------------------------
_SCENIC_KEYWORDS = [
    "peak", "summit", "ridge", "ridgeline", "lookout", "vista", "overlook",
    "panorama", "view", "views", "scenic", "canyon", "falls", "waterfall",
    "creek", "lake", "reservoir", "meadow", "bluff", "point",
]


def _infer_scenic_likelihood(route: Route) -> float:
    """
    Derive a scenic likelihood estimate from existing route attributes.
    Returns 0..1.
    """
    miles = max(0.1, float(route.distance_miles))
    elev = max(0.0, float(route.elevation_gain))
    gpm = elev / miles

    gpm_norm = _clamp((gpm - 100.0) / 600.0, 0.0, 1.0)
    pop_norm = _clamp(float(route.popularity) / 100.0, 0.0, 1.0)

    name = _norm_text(route.name)
    kw_hits = 0
    for kw in _SCENIC_KEYWORDS:
        if kw in name:
            kw_hits += 1
    kw_bonus = _clamp(0.10 * kw_hits, 0.0, 0.35)

    rt = _norm_text(route.route_type)
    loop_bonus = 0.05 if ("loop" in rt) else 0.0

    diff = _norm_text(route.difficulty)
    tech = _norm_text(route.technicality)
    hard_bonus = 0.03 if any(x in diff for x in ["hard", "difficult"]) else 0.0
    tech_bonus = 0.03 if any(x in tech for x in ["technical", "rocky"]) else 0.0

    inferred = (
        0.15
        + 0.50 * gpm_norm
        + 0.20 * pop_norm
        + kw_bonus
        + loop_bonus
        + hard_bonus
        + tech_bonus
    )
    return _clamp(inferred, 0.0, 1.0)


def _effective_scenic(route: Route) -> float:
    """
    Use the best available scenic value:
    - keep dataset value if it's good
    - otherwise "rescue" using inference
    We never reduce scenic likelihood.
    """
    try:
        base = _clamp(float(route.scenic_likelihood), 0.0, 1.0)
    except Exception:
        base = 0.0
    inferred = _infer_scenic_likelihood(route)
    return max(base, inferred)


# ------------------------------------------------------------
# Difficulty scoring (NEW — 1b)
# ------------------------------------------------------------

# Ordered levels for distance-based scoring
_DIFFICULTY_LEVELS = ["easy", "moderate", "hard", "very hard"]


def _difficulty_index(level: str) -> int:
    """Map difficulty string to numeric index. Returns -1 if unknown."""
    normalized = _norm_text(level)
    for i, lvl in enumerate(_DIFFICULTY_LEVELS):
        if lvl in normalized or normalized in lvl:
            return i
    # Common synonyms
    if any(w in normalized for w in ["beginner", "simple", "flat", "gentle"]):
        return 0
    if any(w in normalized for w in ["intermediate", "medium"]):
        return 1
    if any(w in normalized for w in ["challenging", "strenuous", "difficult"]):
        return 2
    if any(w in normalized for w in ["extreme", "expert"]):
        return 3
    return -1


def _difficulty_score(route: Route, prefs: Dict[str, Any]) -> float:
    """
    Score how well the route difficulty matches the user's requested difficulty.

    Scoring:
      - Exact match: 1.0
      - 1 level away: 0.55
      - 2 levels away: 0.20
      - 3 levels away: -0.10 (actively penalizes)
      - No preference stated: 0.6 (neutral)
      - Unknown route difficulty: 0.5 (slight penalty vs known matches)
    """
    user_diff_pref = prefs.get("difficulty_preference", None)
    if user_diff_pref is None:
        return 0.6  # neutral — no preference expressed

    user_idx = _difficulty_index(str(user_diff_pref))
    route_idx = _difficulty_index(str(route.difficulty))

    if user_idx < 0 or route_idx < 0:
        return 0.5  # can't compare — slightly below neutral

    distance = abs(user_idx - route_idx)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.55
    elif distance == 2:
        return 0.20
    else:
        return -0.10


# ------------------------------------------------------------
# Mileage scoring
# ------------------------------------------------------------

def _mileage_score(distance_miles: float, prefs: Dict[str, Any], relax_level: int = 0) -> float:
    """
    Two-zone mileage scoring.
    Zone 1 (within tol): score 1.0 → 0.60
    Zone 2 (outside tol): score 0.60 → negative
    """
    m = float(distance_miles)
    relax_factor = 1.0 + min(0.50, 0.10 * max(0, relax_level))

    if prefs.get("target_miles") is not None:
        t = float(prefs["target_miles"])
        tol = max(0.75, 0.15 * max(t, 1e-6)) * relax_factor
        deviation = abs(m - t)
        if deviation <= tol:
            return 1.0 - (deviation / tol) * 0.40
        overshoot = deviation - tol
        score = 0.60 - (overshoot / tol) * 0.90
        return _clamp(score, -0.40, 0.60)

    min_m = float(prefs.get("min_mileage", 0.0))
    max_m = float(prefs.get("max_mileage", 100.0))
    if max_m < min_m:
        min_m, max_m = max_m, min_m

    mid = (min_m + max_m) / 2.0
    half = max(0.1, (max_m - min_m) / 2.0)

    if min_m <= m <= max_m:
        deviation = abs(m - mid)
        return 1.0 - (deviation / half) * 0.40

    dist_to_range = (min_m - m) if m < min_m else (m - max_m)
    tol_outside = max(1.0, 0.20 * max(mid, 1.0)) * relax_factor
    score = 0.60 - (dist_to_range / tol_outside) * 0.90
    return _clamp(score, -0.40, 0.60)


# ------------------------------------------------------------
# Elevation scoring
# ------------------------------------------------------------

def _elevation_score(
    distance_miles: float,
    elevation_gain: float,
    prefs: Dict[str, Any],
    relax_level: int = 0,
) -> Tuple[float, float, float]:
    """
    Elevation: "no surprises"
    Returns: (elevation_score, elev_gain_score, steepness_score)
    """
    e = float(elevation_gain)
    m = max(0.1, float(distance_miles))
    gain_per_mile = e / m
    relax_factor = 1.0 + min(0.75, 0.15 * max(0, relax_level))

    max_elev = prefs.get("max_elevation", None)
    target_elev = prefs.get("target_elevation_gain", None)

    if target_elev is not None:
        t = float(target_elev)
        tol = max(150.0, 0.25 * max(t, 1.0)) * relax_factor
        elev_gain_score = _clamp(1.0 - abs(e - t) / tol, 0.0, 1.0)
        expected_gain = t
    elif max_elev is not None:
        me = float(max_elev)
        if me <= 0:
            elev_gain_score = 0.0
            expected_gain = 0.0
        else:
            overshoot = max(0.0, e - me)
            tol = max(150.0, 0.25 * me) * relax_factor
            elev_gain_score = _clamp(1.0 - overshoot / tol, 0.0, 1.0)
            expected_gain = me
    else:
        elev_gain_score = 0.6
        expected_gain = None

    if expected_gain is not None:
        if prefs.get("target_miles") is not None:
            expected_miles = max(1.0, float(prefs["target_miles"]))
        else:
            expected_miles = max(
                1.0,
                (float(prefs.get("min_mileage", 0.0))
                 + float(prefs.get("max_mileage", 100.0))) / 2.0,
            )
        max_ft_per_mile = max(50.0, float(expected_gain) / expected_miles)
        steep_overshoot = max(0.0, gain_per_mile - max_ft_per_mile)
        steep_tol = max(100.0, 0.35 * max_ft_per_mile) * relax_factor
        steepness_score = _clamp(1.0 - steep_overshoot / steep_tol, 0.0, 1.0)
    else:
        steepness_score = 0.6

    elevation_score = 0.65 * elev_gain_score + 0.35 * steepness_score
    return (_clamp(elevation_score, 0.0, 1.0), elev_gain_score, steepness_score)


def _views_score(effective_scenic_likelihood: float, prefs: Dict[str, Any]) -> float:
    v = _clamp(float(effective_scenic_likelihood), 0.0, 1.0)
    vp = _clamp(float(prefs.get("views_preference", 0.5)), 0.0, 1.0)
    return _clamp(1.0 - abs(v - vp), 0.0, 1.0)


def _shade_score(shade_pct: float, prefs: Dict[str, Any]) -> float:
    s = _clamp(float(shade_pct) / 100.0, 0.0, 1.0)
    sp = _clamp(float(prefs.get("shade_preference", 0.5)), 0.0, 1.0)
    return _clamp(1.0 - abs(s - sp), 0.0, 1.0)


def _crowds_score(popularity_0_100: float, prefs: Dict[str, Any]) -> float:
    p = _clamp(float(popularity_0_100) / 100.0, 0.0, 1.0)
    mode = str(prefs.get("crowds_preference", "balanced") or "balanced").strip().lower()
    if mode == "popular":
        return p
    if mode == "secluded":
        return 1.0 - p
    return _clamp(1.0 - 2.0 * abs(p - 0.5), 0.0, 1.0)


def _proximity_soft_score(route_prox: float, max_prox: float) -> float:
    if max_prox <= 0:
        return 0.0
    return _clamp(1.0 - (float(route_prox) / float(max_prox)), 0.0, 1.0)


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in (weights or {}).items() if float(v) >= 0.0}
    total = sum(w.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    if 0.98 <= total <= 1.02:
        return w
    return {k: (v / total) for k, v in w.items()}

# ============================================================
# OSM-DERIVED SCORING DIMENSIONS
# ============================================================

# Surface preference token → set of OSM surface values that satisfy it
_SURFACE_GROUPS = {
    "paved":   {"paved", "asphalt", "concrete", "paving_stones"},
    "dirt":    {"dirt", "earth", "ground", "mud", "unpaved"},
    "gravel":  {"gravel", "fine_gravel", "compacted", "pebblestone"},
    "rocky":   {"rock", "stone", "cobblestone"},
    "any":     set(),  # matches everything
}


def _resolve_effective_surface(route: Route) -> str:
    """
    Blend OSM surface with legacy heuristic. Prefer OSM when it's not 'unknown',
    fall back to the gpx_loader heuristic surface_type.
    """
    if route.osm_surface and route.osm_surface != "unknown":
        return route.osm_surface
    return route.surface_type or "unknown"


def _surface_score(route: Route, prefs: Dict[str, Any]) -> float:
    """
    Score how well the route's surface matches user preference.
    Preference shape: prefs.get('surface_pref') in {'paved','dirt','gravel','rocky','any'} or None.
    No preference → neutral 0.5 (doesn't penalize the route, doesn't reward it either).
    """
    surface_pref = prefs.get("surface_pref")
    if not surface_pref or surface_pref == "any":
        return 0.5

    eff_surface = _resolve_effective_surface(route)
    if eff_surface == "unknown":
        return 0.4  # mild penalty for unknown surface when user has a preference

    accepted = _SURFACE_GROUPS.get(surface_pref, set())
    return 1.0 if eff_surface in accepted else 0.0


def _facilities_score(route: Route, prefs: Dict[str, Any]) -> float:
    """
    Reward routes with parking, restrooms, and water access.
    Preference: prefs.get('wants_facilities') is bool. If False/None, light neutral score.
    """
    wants = bool(prefs.get("wants_facilities", False))
    base = 0.0
    if route.osm_has_trailhead_parking:
        base += 0.35
    if route.osm_restroom_count > 0:
        base += 0.25
    if route.osm_drinking_water_count > 0:
        base += 0.15
    if route.osm_picnic_count > 0:  # picnic tables / picnic sites along route
        base += 0.15
    if route.osm_water_count > 0:  # any water (creeks, etc.) is a small bonus
        base += 0.10
    score = _clamp(base, 0.0, 1.0)
    # If user didn't ask, dampen toward neutral so we don't over-favor facility-rich routes
    if not wants:
        score = 0.3 + 0.4 * score  # squashes range to [0.3, 0.7]
    return score


def _scenic_pois_score(route: Route, prefs: Dict[str, Any]) -> float:
    """
    Score based on count of OSM scenic POIs (peaks, viewpoints, waterfalls) along the route.
    Saturates at 5+ POIs.
    """
    n = int(route.osm_scenic_poi_count or 0)
    return _clamp(n / 5.0, 0.0, 1.0)


def _dog_friendly_score(route: Route, prefs: Dict[str, Any]) -> float:
    """
    Score for dog-friendliness. If user doesn't have a dog, returns neutral 0.5.
    If they do, reward routes where dogs are allowed (route tag or park policy).
    """
    has_dog = bool(prefs.get("has_dog", False))
    if not has_dog:
        return 0.5

    # Explicit positive signals
    if route.osm_dog_allowed is True:
        return 1.0
    if route.osm_park_dog_policy in ("yes", "leashed"):
        return 1.0

    # Explicit negative signals
    if route.osm_dog_allowed is False:
        return 0.0
    if route.osm_park_dog_policy == "no":
        return 0.0

    # No info — small uncertainty penalty
    return 0.4


def _technicality_score(route: Route, prefs: Dict[str, Any]) -> float:
    """
    Score how well the route's technicality matches user preference.
    Preference shape: prefs.get('technicality_pref') in {'low','medium','high'} or None.
    Uses OSM mtb:scale/sac_scale when available, falls back to legacy heuristic technicality.
    """
    pref = prefs.get("technicality_pref")
    if not pref:
        return 0.5  # neutral

    # Resolve effective technicality on a 0-6 scale
    if route.osm_technicality and route.osm_technicality > 0:
        tech_score = route.osm_technicality
    else:
        # Fall back to legacy heuristic: map text to a 0-6 scale
        legacy_map = {
            "non-technical": 1.0, "moderate": 2.5,
            "technical": 4.0, "very technical": 5.5,
        }
        tech_score = legacy_map.get(route.technicality, 2.5)

    # User preference targets
    targets = {"low": 1.5, "medium": 3.0, "high": 5.0}
    target = targets.get(pref, 3.0)

    # Distance from target → score (1.0 if exact match, decays linearly)
    dist = abs(tech_score - target)
    return _clamp(1.0 - (dist / 4.0), 0.0, 1.0)

def _gradient_score(route: Route, prefs: Dict[str, Any]) -> float:
    """Score steepness using real per-segment grade data from GPX."""
    max_gain_m = prefs.get("max_gain_m", None)
    if max_gain_m is None:
        return 0.6

    target_miles = prefs.get("target_miles", None)
    if target_miles and float(target_miles) > 0:
        expected_ft_per_mile = (float(max_gain_m) * 3.28084) / max(1.0, float(target_miles))
        expected_max_grade = (expected_ft_per_mile / 5280.0) * 100.0
    else:
        gain_m = float(max_gain_m)
        if gain_m <= 50:
            expected_max_grade = 1.0
        elif gain_m <= 150:
            expected_max_grade = 3.0
        elif gain_m <= 300:
            expected_max_grade = 5.0
        elif gain_m <= 500:
            expected_max_grade = 8.0
        else:
            expected_max_grade = 12.0

    max_grade = float(route.max_grade_pct or 0.0)
    avg_grade = float(route.avg_grade_pct or 0.0)

    overshoot_max = max(0.0, max_grade - expected_max_grade * 1.5)
    tol_max = max(1.0, expected_max_grade * 0.5)
    max_grade_score = _clamp(1.0 - overshoot_max / tol_max, 0.0, 1.0)

    overshoot_avg = max(0.0, avg_grade - expected_max_grade)
    tol_avg = max(1.0, expected_max_grade * 0.4)
    avg_grade_score = _clamp(1.0 - overshoot_avg / tol_avg, 0.0, 1.0)

    return _clamp(0.55 * max_grade_score + 0.45 * avg_grade_score, 0.0, 1.0)


def _score_band(conformity_score: float) -> Optional[str]:
    s = float(conformity_score)
    if s >= BAND_1_MIN:
        return "band_1"
    if s >= BAND_2_MIN:
        return "band_2"
    if s >= BAND_3_MIN:
        return "band_3"
    return None


def select_routes(
    routes: List[Route],
    preferences: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Hard constraints first, then soft scoring.
    Returns list of dicts:
      {route_id, name, conformity_score, score_band, sub_scores, explanation_bits}
    """
    prefs = dict(preferences or {})
    w = _normalize_weights(DEFAULT_WEIGHTS if weights is None else weights)

    user_location = prefs.get("location")
    max_prox = float(prefs.get("max_proximity", 20.0))
    allowed_surfaces = prefs.get("allowed_surface_types")
    preferred_surface = prefs.get("preferred_surface")
    relax_level = int(prefs.get("relax_level", 0) or 0)

    user_lat = prefs.get("lat", None)
    user_lng = prefs.get("lng", None)
    has_user_location = (user_lat is not None and user_lng is not None)

    prox_overflow = 0.0
    if relax_level >= 2:
        prox_overflow = min(10.0, 2.0 * relax_level)

    # Hard gates
    candidates: List[Route] = []
    
    for r in routes:
        if has_user_location and r._start_point is not None:
            live_prox = haversine_miles((user_lat, user_lng), r._start_point)
        else:
            live_prox = r.proximity_miles
        if live_prox > (max_prox + prox_overflow):
            continue

        if user_location is not None and str(user_location).strip():
            if not _location_match(r.location, str(user_location)):
                continue

        if allowed_surfaces:
            try:
                if r.surface_type not in allowed_surfaces:
                    continue
            except TypeError:
                pass
        elif preferred_surface and str(preferred_surface).strip().lower() != "mixed":
            if r.surface_type != preferred_surface:
                continue

        candidates.append(r)

    if not candidates:
        return []

    # Intent gate — loop / out-and-back hard filter
    intent = prefs.get("intent", None)
    if intent:
        intent_lower = intent.strip().lower()
        if intent_lower in ("loop", "out-and-back"):
            intent_filtered = [r for r in candidates if r.route_type == intent_lower]
            if intent_filtered:
                candidates = intent_filtered

    # Hard distance pre-filter — tiered window based on target distance
    target_miles_val = prefs.get("target_miles", None)
    if target_miles_val is not None:
        t = float(target_miles_val)
        if t < 4.0:
            window_pct = 0.25
        elif t < 10.0:
            window_pct = 0.30
        else:
            window_pct = 0.40
        if relax_level >= 2:
            window_pct += 0.10
        lower_bound = t * (1.0 - window_pct)
        upper_bound = t * (1.0 + window_pct)
        filtered = [r for r in candidates if lower_bound <= r.distance_miles <= upper_bound]
        if filtered:
            candidates = filtered

    # Bounding-box geographic filter
    bbox_min_lat = prefs.get("bbox_min_lat")
    bbox_min_lng = prefs.get("bbox_min_lng")
    bbox_max_lat = prefs.get("bbox_max_lat")
    bbox_max_lng = prefs.get("bbox_max_lng")
    has_bbox = all(v is not None for v in [bbox_min_lat, bbox_min_lng, bbox_max_lat, bbox_max_lng])
    if has_bbox:
        bbox_min_lat = float(bbox_min_lat)
        bbox_min_lng = float(bbox_min_lng)
        bbox_max_lat = float(bbox_max_lat)
        bbox_max_lng = float(bbox_max_lng)
        bbox_filtered = [
            r for r in candidates
            if r._start_point is None or (
                bbox_min_lat <= r._start_point[0] <= bbox_max_lat
                and bbox_min_lng <= r._start_point[1] <= bbox_max_lng
            )
        ]
        if bbox_filtered:
            candidates = bbox_filtered

    # ============================================================
    # OSM HARD GATES (absolute filters, not weighted)
    # ============================================================

    # Bike-legal gate: exclude routes where OSM marks bicycles=no
    if prefs.get("require_bike_legal"):
        candidates = [r for r in candidates if r.osm_bicycle_legal]

    # Dog-required gate: exclude routes that prohibit dogs
    if prefs.get("require_dog_allowed"):
        candidates = [
            r for r in candidates
            if r.osm_dog_allowed is not False
            and r.osm_park_dog_policy != "no"
        ]

    # Wheelchair / accessibility gate: paved surfaces only
    if prefs.get("require_wheelchair_accessible"):
        ACCESSIBLE_SURFACES = {"paved", "asphalt", "concrete", "paving_stones"}
        candidates = [
            r for r in candidates
            if (r.osm_surface in ACCESSIBLE_SURFACES) or (r.surface_type == "paved")
        ]

    if not candidates:
        return []

    views_pref = _clamp(float(prefs.get("views_preference", 0.5)), 0.0, 1.0)
    scenic_offsets_elev = (views_pref >= 0.6)
    scenic_bonus_factor = 0.25

    results: List[Dict[str, Any]] = []

    for r in candidates:
        sub: Dict[str, float] = {}

        if has_user_location and r._start_point is not None:
            live_prox = haversine_miles((user_lat, user_lng), r._start_point)
        else:
            live_prox = r.proximity_miles

        scenic_eff = _effective_scenic(r)

        sub["mileage"] = _mileage_score(r.distance_miles, prefs, relax_level=relax_level)

        elev_score, elev_gain_score, steep_score = _elevation_score(
            r.distance_miles, r.elevation_gain, prefs, relax_level=relax_level
        )
        gradient = _gradient_score(r, prefs)
        new_elev_score = _clamp(0.65 * elev_gain_score + 0.35 * gradient, 0.0, 1.0)
        if scenic_offsets_elev:
            new_elev_score = _clamp(
                new_elev_score + scenic_bonus_factor * _clamp(scenic_eff, 0.0, 1.0),
                0.0, 1.0,
            )
        new_elev_score = _clamp(
                new_elev_score + scenic_bonus_factor * _clamp(scenic_eff, 0.0, 1.0),
                0.0, 1.0,
            )

        sub["elevation"] = new_elev_score
        sub["elev_gain"] = _clamp(elev_gain_score, 0.0, 1.0)
        sub["steepness"] = gradient
        sub["views"] = _views_score(scenic_eff, prefs)
        sub["views"] = _views_score(scenic_eff, prefs)
        # Use OSM shade if available, fall back to legacy heuristic shade_pct
        effective_shade = r.osm_shade_pct if r.osm_shade_pct > 0 else r.shade_pct
        sub["shade"] = _shade_score(effective_shade, prefs)
        sub["proximity"] = _proximity_soft_score(live_prox, max_prox)
        sub["crowds"] = _crowds_score(r.popularity, prefs)
        sub["difficulty"] = _difficulty_score(r, prefs)

        # ----------------------------------------------------------
        # OSM-driven sub-scores
        # ----------------------------------------------------------
        sub["surface"]      = _surface_score(r, prefs)
        sub["facilities"]   = _facilities_score(r, prefs)
        sub["scenic_pois"]  = _scenic_pois_score(r, prefs)
        sub["dog_friendly"] = _dog_friendly_score(r, prefs)
        sub["technicality"] = _technicality_score(r, prefs)

        score_0_1 = 0.0
        score_0_1 += w.get("mileage", 0.0)      * sub["mileage"]
        score_0_1 += w.get("elevation", 0.0)    * sub["elevation"]
        score_0_1 += w.get("views", 0.0)        * sub["views"]
        score_0_1 += w.get("shade", 0.0)        * sub["shade"]
        score_0_1 += w.get("proximity", 0.0)    * sub["proximity"]
        score_0_1 += w.get("crowds", 0.0)       * sub["crowds"]
        score_0_1 += w.get("difficulty", 0.0)   * sub["difficulty"]
        score_0_1 += w.get("surface", 0.0)      * sub["surface"]
        score_0_1 += w.get("facilities", 0.0)   * sub["facilities"]
        score_0_1 += w.get("scenic_pois", 0.0)  * sub["scenic_pois"]
        score_0_1 += w.get("dog_friendly", 0.0) * sub["dog_friendly"]
        score_0_1 += w.get("technicality", 0.0) * sub["technicality"]

        conformity = round(_clamp(score_0_1, 0.0, 1.0) * 100.0, 2)
        band = _score_band(conformity)
        if band is None:
            continue

        # ============================================================
        # Explanation bits (2–3 max)
        # ============================================================
        explain: List[str] = []

        if live_prox <= max_prox:
            explain.append("close by" if sub["proximity"] >= 0.75 else "nearby")
        else:
            explain.append("a bit farther")

        if sub["mileage"] >= 0.85:
            explain.append("mileage match")
        elif sub["mileage"] >= 0.65:
            explain.append(f"about {round(r.distance_miles, 1)} miles")
        else:
            if relax_level >= 2:
                explain.append("slightly outside target")

        if sub["elevation"] >= 0.85:
            explain.append("low elevation (no surprise)")
        else:
            if prefs.get("max_elevation") is not None:
                if sub["elev_gain"] < 0.55 or sub["steepness"] < 0.55:
                    explain.append("a bit more climb" if relax_level >= 2 else "more climb")

        # Difficulty explanation
        if sub["difficulty"] >= 0.90:
            explain.append(f"{r.difficulty} trail")
        elif sub["difficulty"] <= 0.25:
            user_diff = prefs.get("difficulty_preference", "")
            explain.append(f"harder than {user_diff}" if user_diff else "harder trail")

        if views_pref >= 0.7 and _clamp(scenic_eff, 0.0, 1.0) >= 0.7:
            explain.append("big views")

        shade_pref = _clamp(float(prefs.get("shade_preference", 0.5)), 0.0, 1.0)
        if shade_pref >= 0.7 and _clamp(r.shade_pct / 100.0, 0.0, 1.0) >= 0.7:
            explain.append("shady canopy")

        crowd_mode = str(prefs.get("crowds_preference", "balanced") or "balanced").strip().lower()
        p_norm = _clamp(r.popularity / 100.0, 0.0, 1.0)
        if crowd_mode == "secluded" and p_norm <= 0.3:
            explain.append("quiet trail")
        elif crowd_mode == "popular" and p_norm >= 0.7:
            explain.append("popular classic")

        explain_unique: List[str] = []
        for ebit in explain:
            if ebit not in explain_unique:
                explain_unique.append(ebit)
            if len(explain_unique) >= 3:
                break

        results.append({
            "route_id":        r.route_id,
            "name":            r.name,
            "route_type":      r.route_type,
            "max_grade_pct":   r.max_grade_pct,
            "avg_grade_pct":   r.avg_grade_pct,
            "conformity_score": conformity,
            "score_band":      band,
            "sub_scores":      sub,
            "explanation_bits": explain_unique,
        })

    results.sort(key=lambda x: x["conformity_score"], reverse=True)
    return results

