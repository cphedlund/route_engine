from __future__ import annotations

import contextlib
import io
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

TOP_K = 5

# A route is "flat" for benchmark purposes when total gain is
# <= max(FLAT_MAX_GAIN_FT, FLAT_GAIN_FT_PER_MILE x the route's own distance).
# 60 ft/mi is ~1.1% average grade, where most hiking guides stop calling a route
# flat. The 300 ft floor keeps short routes from being failed for GPS elevation
# noise (it is also above the engine's own flat cap, the library's 25th-percentile
# gain of 212 ft). Scaling by distance keeps the rule fair for longer flat
# queries: 300 ft at 5 mi, 480 ft at 8 mi. The reported bug returned ~2000 ft.
# Cases opt in with `elev_gain_ft_max: FLAT`.
FLAT_MAX_GAIN_FT = 300.0
FLAT_GAIN_FT_PER_MILE = 60.0


def flat_max_gain_ft(distance_mi: float) -> float:
    return max(FLAT_MAX_GAIN_FT, FLAT_GAIN_FT_PER_MILE * float(distance_mi))


NAMED_CONSTANTS = {
    "FLAT_MAX_GAIN_FT": FLAT_MAX_GAIN_FT,
}

EXPLANATION_KEYS = ("message", "explanation", "relaxation_note", "notice")

PAVED_SURFACES = {"paved", "asphalt", "concrete", "paving_stones"}

# Wheelchair ground truth is deliberately stricter than the engine gate. The
# engine accepts a route if EITHER osm_surface is paved OR the GPX heuristic
# surface_type is "paved". A wheelchair user is harmed by a false positive, so
# the benchmark requires the heuristic to say paved AND OSM to not contradict it
# (osm_surface paved or unknown). 45 routes are heuristic-paved but OSM-unpaved.

_ENGINE: Dict[str, Any] = {}


def _engine():
    if not _ENGINE:
        with contextlib.redirect_stdout(io.StringIO()):
            import app as appmod
        appmod.LLM_TRANSLATION_ENABLED = os.environ.get("ATLAS_BENCH_LLM") == "1"
        _ENGINE["app"] = appmod
        _ENGINE["routes"] = {r.route_id: r for r in appmod.ROUTE_DB}
    return _ENGINE


def all_routes() -> List[Any]:
    return list(_engine()["routes"].values())


def route_by_id(rid: str):
    return _engine()["routes"].get(rid)


def load_cases() -> List[Dict[str, Any]]:
    with open(CASES_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["cases"] if isinstance(data, dict) else data


def _resolve(v: Any) -> Any:
    if isinstance(v, str) and v in NAMED_CONSTANTS:
        return NAMED_CONSTANTS[v]
    return v


def route_facts(r) -> Dict[str, Any]:
    return {
        "route_id": r.route_id,
        "name": r.name,
        "distance_mi": round(float(r.distance_miles), 2),
        "elev_gain_ft": round(float(r.elevation_gain), 0),
        "park": r.osm_park_name,
        "route_type": r.route_type,
        "surface_type": r.surface_type,
        "bike_legal": bool(r.osm_bicycle_legal),
        "dog_ok": (r.osm_dog_allowed is not False and r.osm_park_dog_policy != "no"),
        "paved": (r.surface_type == "paved") and (r.osm_surface in PAVED_SURFACES or r.osm_surface == "unknown"),
        "start": (r.start_lat, r.start_lng) if r.start_lat is not None and r.start_lng is not None else None,
    }


def _prox_mi(facts: Dict[str, Any], user: Optional[Dict[str, float]]) -> Optional[float]:
    if not user or not facts.get("start"):
        return None
    from gpx_loader import haversine_miles
    return haversine_miles((user["lat"], user["lng"]), facts["start"])


# Violation rule. Categorical and safety gates (bike, dog, wheelchair, park,
# route_type, proximity, must_exclude_where) may never be violated by any returned
# route. Numeric targets (distance, elevation) follow a feasible-first rule: with F
# routes satisfying every hard constraint, the first F results must all satisfy
# them; results ranked after F are filler and may miss numeric targets.
def hard_violations(facts: Dict[str, Any], hard: Dict[str, Any], user: Optional[Dict[str, float]] = None) -> List[str]:
    out: List[str] = []
    wp = hard.get("within_mi_of_user")
    if wp is not None:
        d = _prox_mi(facts, user)
        if d is None or d > float(wp):
            out.append(f"gate: start {None if d is None else round(d, 1)} mi from user, limit {wp}")
    dm = hard.get("distance_mi")
    if dm and not (float(dm[0]) <= facts["distance_mi"] <= float(dm[1])):
        out.append(f"soft: distance {facts['distance_mi']} outside {dm}")
    eg = hard.get("elev_gain_ft_max")
    eg = flat_max_gain_ft(facts["distance_mi"]) if eg == "FLAT" else _resolve(eg)
    if eg is not None and facts["elev_gain_ft"] > float(eg):
        out.append(f"soft: gain {facts['elev_gain_ft']:.0f} > {float(eg):.0f}")
    eg_min = hard.get("elev_gain_ft_min")
    if eg_min is not None and facts["elev_gain_ft"] < float(eg_min):
        out.append(f"soft: gain {facts['elev_gain_ft']:.0f} < {float(eg_min):.0f}")
    if hard.get("require_bike_legal") and not facts["bike_legal"]:
        out.append("gate: not bike legal")
    if hard.get("require_dog_allowed") and not facts["dog_ok"]:
        out.append("gate: dogs not allowed")
    if hard.get("require_wheelchair") and not facts["paved"]:
        out.append("gate: not paved/wheelchair accessible")
    if hard.get("park") and facts["park"] != hard["park"]:
        out.append(f"gate: park {facts['park']!r} != {hard['park']!r}")
    if hard.get("route_type") and facts["route_type"] != hard["route_type"]:
        out.append(f"gate: route_type {facts['route_type']} != {hard['route_type']}")
    return out


def exclude_violation(facts: Dict[str, Any], expr: Optional[str]) -> bool:
    if not expr:
        return False
    return bool(eval(expr, {"__builtins__": {}}, dict(facts)))


def feasible_set(hard: Dict[str, Any], user: Optional[Dict[str, float]] = None) -> List[str]:
    return [r.route_id for r in all_routes() if not hard_violations(route_facts(r), hard, user)]


_CACHE: Dict[str, Dict[str, Any]] = {}


PAGE_BATCH = 3


def run_query(query: str, preferences: Optional[Dict[str, Any]] = None, pages: int = 1) -> Dict[str, Any]:
    key = query + "||" + repr(sorted((preferences or {}).items(), key=lambda kv: kv[0])) + f"||{pages}"
    if key in _CACHE:
        return _CACHE[key]
    eng = _engine()
    appmod = eng["app"]
    result: Dict[str, Any] = {"query": query, "error": None, "routes": [], "prefs": {}, "response_keys": []}
    t0 = time.perf_counter()
    try:
        body = appmod.StartSearchBody(
            query=query,
            preferences=appmod.Preferences(**(preferences or {})),
            batch_size=TOP_K if pages == 1 else PAGE_BATCH,
            new_search=True,
        )
        resp = appmod._start_search_core(body)
        extra_items: List[Dict[str, Any]] = []
        sid = resp["session_id"]
        for _ in range(pages - 1):
            more = appmod.more_results(appmod.MoreResultsIn(session_id=sid, n=PAGE_BATCH), None)
            sid = more["session_id"]
            extra_items.extend(more.get("routes", []))
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        result["response_keys"] = sorted(resp.keys())
        result["explanation"] = next((resp[k] for k in EXPLANATION_KEYS if resp.get(k)), None)
        try:
            result["prefs"] = appmod.read_session_token(resp["session_id"])["prefs"]
        except Exception:
            result["prefs"] = {}
        for item in list(resp.get("routes", [])) + extra_items:
            r = eng["routes"].get(item["route_id"])
            facts = route_facts(r) if r is not None else {"route_id": item["route_id"]}
            facts["score"] = item.get("conformity_score")
            result["routes"].append(facts)
    except Exception as exc:
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        result["error"] = f"{type(exc).__name__}: {exc}"
    _CACHE[key] = result
    return result


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    expect = case.get("expect", {}) or {}
    hard = expect.get("hard", {}) or {}
    excl = expect.get("must_exclude_where")
    infeasible = bool(expect.get("infeasible", False))
    prefs_in = case.get("preferences") or {}
    user = case.get("user_location")
    if user is None and "lat" in prefs_in and "lng" in prefs_in:
        user = {"lat": prefs_in["lat"], "lng": prefs_in["lng"]}
    pages = int(expect.get("pages", 1))
    res = run_query(case["query"], prefs_in, pages)
    routes = res["routes"]

    feasible_ids = feasible_set(hard, user) if hard else [r.route_id for r in all_routes()]
    feasible = len(feasible_ids) > 0 and not infeasible

    failures: List[str] = []
    violations: List[Dict[str, Any]] = []

    if res["error"]:
        failures.append(f"crash: {res['error']}")

    if not infeasible:
        for idx, f in enumerate(routes):
            v = hard_violations(f, hard, user) if hard else []
            if idx >= len(feasible_ids):
                v = [x for x in v if x.startswith("gate:")]
            if exclude_violation(f, excl):
                v.append(f"gate: matches must_exclude_where: {excl}")
            if v:
                violations.append({"route_id": f["route_id"], "distance_mi": f.get("distance_mi"),
                                   "elev_gain_ft": f.get("elev_gain_ft"), "reasons": v})
        if violations:
            failures.append(f"{len(violations)}/{len(routes)} returned routes violate hard constraints")

    empty_when_feasible = (not res["error"]) and feasible and len(routes) == 0
    if expect.get("non_empty") and feasible and len(routes) == 0:
        failures.append(f"empty result but {len(feasible_ids)} feasible routes exist")

    hit: Optional[bool] = None
    mia = expect.get("must_include_any")
    if mia is not None and not infeasible:
        target = set(feasible_ids) if mia == "auto" else set(mia)
        if target:
            hit = any(f["route_id"] in target for f in routes[:TOP_K])
            if not hit:
                failures.append("no must_include_any route in top 5")

    if infeasible:
        if feasible_ids:
            failures.append(f"case marked infeasible but {len(feasible_ids)} routes satisfy hard constraints (bad case)")
        if not res["error"] and not (routes and res.get("explanation")):
            failures.append("infeasible query: expected relaxed results plus an explanation field "
                            f"(got {len(routes)} routes, explanation={res.get('explanation')!r}, keys={res['response_keys']})")

    if expect.get("no_crash_only"):
        failures = [f for f in failures if f.startswith("crash")]

    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "tags": case.get("tags", []) or [],
        "query": case["query"],
        "owner": case.get("owner"),
        "passed": not failures,
        "failures": failures,
        "violations": violations,
        "violating": bool(violations),
        "violation_applicable": (not infeasible) and (bool(hard) or bool(excl)),
        "empty_when_feasible": empty_when_feasible,
        "feasible": feasible,
        "feasible_count": len(feasible_ids),
        "hit": hit,
        "latency_ms": res.get("latency_ms", 0.0),
        "prefs": res["prefs"],
        "top": routes,
        "error": res["error"],
    }


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    app_cases = [r for r in results if r["violation_applicable"]]
    feas_cases = [r for r in results if r["feasible"]]
    hit_cases = [r for r in results if r["hit"] is not None]
    lat = [r["latency_ms"] for r in results]
    return {
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "failed": sum(not r["passed"] for r in results),
        "constraint_violation_rate": (sum(r["violating"] for r in app_cases) / len(app_cases)) if app_cases else 0.0,
        "violation_cases": f"{sum(r['violating'] for r in app_cases)}/{len(app_cases)}",
        "empty_when_feasible_rate": (sum(r["empty_when_feasible"] for r in feas_cases) / len(feas_cases)) if feas_cases else 0.0,
        "empty_cases": f"{sum(r['empty_when_feasible'] for r in feas_cases)}/{len(feas_cases)}",
        "hit_rate_at_5": (sum(bool(r["hit"]) for r in hit_cases) / len(hit_cases)) if hit_cases else 0.0,
        "hit_cases": f"{sum(bool(r['hit']) for r in hit_cases)}/{len(hit_cases)}",
        "latency_p50_ms": round(statistics.median(lat), 1) if lat else 0.0,
        "latency_p95_ms": round(_pct(lat, 0.95), 1),
    }


def by_category(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cats: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    return {k: summarize(v) for k, v in sorted(cats.items())}


def run_all() -> List[Dict[str, Any]]:
    return [evaluate_case(c) for c in load_cases()]
