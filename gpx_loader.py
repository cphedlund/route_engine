"""
GPX loader for the route engine.

Canonical units
- distance_miles: statute miles (haversine, R = 3958.7613 mi)
- elevation_gain, osm elevations, GPX <ele>: GPX elevations are metres; derived gain is stored in feet
- grades: percent (rise / run x 100)
- coordinates: WGS84 decimal degrees, (lat, lng) tuples internally

Validation (see validate_track): tracks failing a rule are rejected, logged with a reason,
and counted in LAST_LOAD_REPORT. Rejected files never reach the route list.
"""
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import osm_layers

import gpxpy


# ============================================================
# Geometry helpers
# ============================================================

def haversine_miles(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 3958.7613  # miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def _extract_points(gpx) -> List[Tuple[float, float, Optional[float]]]:
    pts: List[Tuple[float, float, Optional[float]]] = []
    for track in getattr(gpx, "tracks", []):
        for seg in getattr(track, "segments", []):
            for p in getattr(seg, "points", []):
                pts.append((p.latitude, p.longitude, p.elevation))
    if pts:
        return pts
    # Fallback: route points
    for rte in getattr(gpx, "routes", []):
        for p in getattr(rte, "points", []):
            pts.append((p.latitude, p.longitude, getattr(p, "elevation", None)))
    return pts


# ============================================================
# Validation
# ============================================================

MIN_POINTS = 10
MAX_JUMP_M = 500.0
MAX_ROUTE_MILES = 100.0
MIN_ROUTE_MILES = 0.05
MIN_ELEVATION_COVERAGE = 0.9
ELEVATION_RANGE_M = (-450.0, 9000.0)
SCC_BBOX = (36.89, 37.49, -122.21, -121.20)
REGION_PAD_DEG = 0.25
OUT_OF_COUNTY_POLICY = "flag"

LAST_LOAD_REPORT: Dict[str, Any] = {}


def _record(bucket: List[Dict[str, Any]], filename: str, **fields: Any) -> None:
    entry = {"file": filename}
    entry.update(fields)
    bucket.append(entry)


def _in_bbox(lat: float, lng: float, bbox: Tuple[float, float, float, float], pad: float = 0.0) -> bool:
    return (bbox[0] - pad <= lat <= bbox[1] + pad) and (bbox[2] - pad <= lng <= bbox[3] + pad)


def validate_track(pts: List[Tuple[float, float, Optional[float]]]) -> Dict[str, Any]:
    """
    Returns {"ok": bool, "reason": str, "detail": str, "warnings": [str], "distance_miles": float}.
    Hard rejections: too_few_points, nan_coordinate, bad_coordinate, missing_elevation,
    bad_elevation, teleport_jump, outside_region, zero_length, impossible_distance.
    Soft flag: outside_county (inside the padded region but outside SCC_BBOX) when
    OUT_OF_COUNTY_POLICY == "flag"; rejected when it is "reject".
    """
    def fail(reason: str, detail: str = "") -> Dict[str, Any]:
        return {"ok": False, "reason": reason, "detail": detail, "warnings": [], "distance_miles": 0.0}

    if len(pts) < MIN_POINTS:
        return fail("too_few_points", f"{len(pts)} < {MIN_POINTS}")

    for lat, lng, ele in pts:
        if lat is None or lng is None or not (math.isfinite(lat) and math.isfinite(lng)):
            return fail("nan_coordinate")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
            return fail("bad_coordinate", f"{lat},{lng}")

    with_ele = [p[2] for p in pts if p[2] is not None]
    if any(not math.isfinite(e) for e in with_ele):
        return fail("bad_elevation", "non-finite elevation")
    if len(with_ele) / len(pts) < MIN_ELEVATION_COVERAGE:
        return fail("missing_elevation", f"{len(with_ele)}/{len(pts)} points have elevation")
    if any(not (ELEVATION_RANGE_M[0] <= e <= ELEVATION_RANGE_M[1]) for e in with_ele):
        return fail("bad_elevation", f"outside {ELEVATION_RANGE_M} m")

    warnings: List[str] = []
    outside_county = [p for p in pts if not _in_bbox(p[0], p[1], SCC_BBOX)]
    if outside_county:
        if any(not _in_bbox(p[0], p[1], SCC_BBOX, REGION_PAD_DEG) for p in pts):
            return fail("outside_region", "points beyond Santa Clara County bbox + padding")
        if OUT_OF_COUNTY_POLICY == "reject":
            return fail("outside_county", f"{len(outside_county)} points outside SCC bbox")
        warnings.append("outside_county")

    total = 0.0
    worst = 0.0
    for a, b in zip(pts, pts[1:]):
        d_mi = haversine_miles((a[0], a[1]), (b[0], b[1]))
        total += d_mi
        worst = max(worst, d_mi * 1609.344)
    if worst > MAX_JUMP_M:
        return fail("teleport_jump", f"max consecutive gap {worst:.0f} m > {MAX_JUMP_M:.0f} m")
    if total < MIN_ROUTE_MILES:
        return fail("zero_length", f"{total:.3f} mi")
    if total > MAX_ROUTE_MILES:
        return fail("impossible_distance", f"{total:.1f} mi > {MAX_ROUTE_MILES:.0f} mi")

    return {"ok": True, "reason": "", "detail": "", "warnings": warnings, "distance_miles": total}


# ============================================================
# Stable route ID generation
# ============================================================

def stable_route_id(path: Path, pts: List[Tuple[float, float, Optional[float]]]) -> str:
    """
    Generates a stable, deterministic route_id that:
    - Does NOT depend on absolute filesystem paths
    - Is stable across restarts
    - Is extremely unlikely to collide
    """
    h = hashlib.sha1()
    h.update(path.stem.encode("utf-8"))
    # Use a sparse sampling of points for stability
    step = max(1, len(pts) // 25)
    for i in range(0, len(pts), step):
        lat, lon, _ = pts[i]
        h.update(f"{lat:.5f},{lon:.5f}".encode("utf-8"))
    return h.hexdigest()[:16]  # short, stable, URL-safe


def normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


# ============================================================
# Metadata enrichment helpers (NEW)
# ============================================================

_SCENIC_KEYWORDS = [
    "peak", "summit", "ridge", "ridgeline", "lookout", "vista", "overlook",
    "panorama", "view", "views", "scenic", "canyon", "falls", "waterfall",
    "creek", "lake", "reservoir", "meadow", "bluff", "point", "crest",
    "knoll", "saddle", "pass", "basin", "gorge", "cliff", "butte",
]

_TECHNICAL_KEYWORDS = [
    "scramble", "rock", "rocky", "boulder", "bouldering", "talus",
    "scree", "exposed", "knife", "class", "technical", "steep",
    "chimney", "rappel", "chain", "ladder", "switchback",
]

_PAVED_KEYWORDS = [
    "road", "street", "boulevard", "blvd", "avenue", "ave", "highway",
    "hwy", "path", "bike path", "bikeway", "paved", "sidewalk",
    "boardwalk", "promenade", "greenway",
]

_DIRT_KEYWORDS = [
    "trail", "hike", "hiking", "wilderness", "backcountry", "fire road",
    "fireroad", "singletrack", "single track", "dirt", "canyon",
    "ridge", "peak", "summit", "mountain", "mt", "gulch",
]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _infer_difficulty(
    gain_ft: float,
    distance_miles: float,
    max_grade_pct: float,
    avg_grade_pct: float,
    elev_range_ft: float,
) -> str:
    """
    Derive difficulty from measurable GPX characteristics.

    Factors:
      - Gain per mile (sustained effort)
      - Max grade (steepest section)
      - Total elevation range (exposure/altitude factor)
      - Distance (longer = harder at same intensity)
    """
    miles = max(0.1, distance_miles)
    gpm = gain_ft / miles  # ft gained per mile

    # Build a composite score 0..1
    # Gain-per-mile: 0-100=easy, 100-300=moderate, 300-500=hard, 500+=very hard
    gpm_score = _clamp((gpm - 50.0) / 500.0, 0.0, 1.0)

    # Max grade: under 8% is easy, 8-15% moderate, 15-25% hard, 25%+ very hard
    grade_score = _clamp((max_grade_pct - 5.0) / 25.0, 0.0, 1.0)

    # Distance factor: longer routes are inherently harder
    # Under 3 mi = easy, 3-8 mi = moderate, 8-15 mi = hard, 15+ = very hard
    dist_score = _clamp((miles - 2.0) / 15.0, 0.0, 1.0)

    # Elevation range: routes with big total vertical span are harder
    range_score = _clamp((elev_range_ft - 200.0) / 3000.0, 0.0, 1.0)

    composite = (
        0.40 * gpm_score
        + 0.25 * grade_score
        + 0.20 * dist_score
        + 0.15 * range_score
    )

    if composite < 0.20:
        return "easy"
    elif composite < 0.45:
        return "moderate"
    elif composite < 0.70:
        return "hard"
    else:
        return "very hard"


def _infer_technicality(
    name: str,
    max_grade_pct: float,
    avg_grade_pct: float,
    grade_variance: float,
) -> str:
    """
    Estimate technicality from grade variability and name keywords.

    High grade variance = terrain is inconsistent (rocks, steps, scrambles).
    Low grade variance + low avg = smooth, groomed trail or road.
    """
    name_lower = name.lower()
    kw_hits = sum(1 for kw in _TECHNICAL_KEYWORDS if kw in name_lower)

    # Grade variance: smooth trails have low variance, technical ones have high
    # Typical ranges: 0-3 = smooth, 3-8 = moderate, 8+ = rough
    var_score = _clamp((grade_variance - 2.0) / 10.0, 0.0, 1.0)

    # Max grade spikes indicate technical sections
    spike_score = _clamp((max_grade_pct - 12.0) / 20.0, 0.0, 1.0)

    # Keyword bonus
    kw_bonus = _clamp(0.15 * kw_hits, 0.0, 0.40)

    composite = 0.45 * var_score + 0.30 * spike_score + kw_bonus

    if composite < 0.20:
        return "non-technical"
    elif composite < 0.45:
        return "moderate"
    elif composite < 0.70:
        return "technical"
    else:
        return "very technical"


def _infer_surface(name: str, avg_grade_pct: float, gain_per_mile: float) -> str:
    """
    Rough surface estimate from name keywords and terrain characteristics.

    Paved routes: lower grades, keywords like "road", "path", "bike"
    Dirt routes: higher grades, keywords like "trail", "peak", "ridge"
    Mixed: ambiguous signals
    """
    name_lower = name.lower()
    paved_hits = sum(1 for kw in _PAVED_KEYWORDS if kw in name_lower)
    dirt_hits = sum(1 for kw in _DIRT_KEYWORDS if kw in name_lower)

    # Terrain signal: steeper and more gain → more likely dirt/trail
    terrain_dirt_signal = _clamp((gain_per_mile - 80.0) / 300.0, 0.0, 1.0)

    paved_score = paved_hits * 0.3 + (1.0 - terrain_dirt_signal) * 0.3
    dirt_score = dirt_hits * 0.3 + terrain_dirt_signal * 0.4

    if paved_score > dirt_score + 0.15:
        return "paved"
    elif dirt_score > paved_score + 0.15:
        return "dirt"
    else:
        return "mixed"


def _infer_scenic_likelihood(
    name: str,
    gain_per_mile: float,
    elev_range_ft: float,
    max_elev_ft: float,
    route_type: str,
) -> float:
    """
    Estimate scenic likelihood from GPX-derivable attributes.

    Signals:
      - Elevation gain per mile (proxy for getting to viewpoints)
      - Maximum elevation (higher = more likely to have views)
      - Elevation range (big range = likely ridge/valley contrast)
      - Name keywords ("peak", "vista", "canyon", etc.)
      - Route type (loops slightly favored — designed experiences)
    """
    # Gain per mile: ~100-900 ft/mi range
    gpm_norm = _clamp((gain_per_mile - 80.0) / 500.0, 0.0, 1.0)

    # Max elevation: higher routes tend to be more scenic
    # SB/LA area: 0-6000+ ft. Coastal trails at <200 ft can be scenic too.
    elev_norm = _clamp((max_elev_ft - 500.0) / 4000.0, 0.0, 1.0)

    # Elevation range: big range means the route traverses interesting terrain
    range_norm = _clamp((elev_range_ft - 200.0) / 2500.0, 0.0, 1.0)

    # Name keywords
    name_lower = name.lower()
    kw_hits = sum(1 for kw in _SCENIC_KEYWORDS if kw in name_lower)
    kw_bonus = _clamp(0.12 * kw_hits, 0.0, 0.40)

    # Loop bonus
    loop_bonus = 0.05 if route_type == "loop" else 0.0

    inferred = (
        0.10                   # baseline — most trails have some scenic value
        + 0.30 * gpm_norm
        + 0.15 * elev_norm
        + 0.15 * range_norm
        + kw_bonus
        + loop_bonus
    )

    return _clamp(inferred, 0.0, 1.0)


def _compute_grade_variance(grades: List[float]) -> float:
    """Standard deviation of grade percentages — proxy for terrain roughness."""
    if len(grades) < 2:
        return 0.0
    mean = sum(grades) / len(grades)
    variance = sum((g - mean) ** 2 for g in grades) / len(grades)
    return math.sqrt(variance)


# ============================================================
# Loader
# ============================================================

def load_routes_from_gpx_dir(gpx_dir: str) -> List[Dict[str, Any]]:
    base = Path(gpx_dir)
    routes: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    flagged: List[Dict[str, Any]] = []
    scanned = 0

    for path in sorted(base.rglob("*.gpx")):
        scanned += 1
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                gpx = gpxpy.parse(f)

            pts = _extract_points(gpx)
            verdict = validate_track(pts)
            if not verdict["ok"]:
                _record(rejected, path.name, reason=verdict["reason"], detail=verdict["detail"])
                print(f"[GPX] REJECTED {path.name}: {verdict['reason']} ({verdict['detail']})")
                continue
            if verdict["warnings"]:
                _record(flagged, path.name, warnings=verdict["warnings"])
                print(f"[GPX] FLAGGED {path.name}: {','.join(verdict['warnings'])}")

            # -------------------------
            # Distance + gradient profile
            # -------------------------
            dist = 0.0
            grades: List[float] = []
            for i in range(1, len(pts)):
                seg_dist_miles = haversine_miles(
                    (pts[i - 1][0], pts[i - 1][1]),
                    (pts[i][0],     pts[i][1]),
                )
                dist += seg_dist_miles
                seg_dist_ft = seg_dist_miles * 5280.0
                if (seg_dist_ft > 5.0
                        and pts[i - 1][2] is not None
                        and pts[i][2] is not None):
                    elev_delta_ft = (pts[i][2] - pts[i - 1][2]) * 3.28084
                    grade_pct = (elev_delta_ft / seg_dist_ft) * 100.0
                    grades.append(abs(grade_pct))

            max_grade_pct = min(100.0, max(grades)) if grades else 0.0
            avg_grade_pct = min(100.0, sum(grades) / len(grades)) if grades else 0.0
            grade_variance = _compute_grade_variance(grades)

            # -------------------------
            # Elevation gain (meters → feet)
            # -------------------------
            gain_ft = 0.0
            last_e = pts[0][2]
            for i in range(1, len(pts)):
                e = pts[i][2]
                if e is None or last_e is None:
                    last_e = e
                    continue
                d = e - last_e
                if d > 0:
                    gain_ft += d * 3.28084
                last_e = e

            # -------------------------
            # Elevation statistics (for enrichment)
            # -------------------------
            elevations_ft = [
                p[2] * 3.28084 for p in pts if p[2] is not None
            ]
            if elevations_ft:
                max_elev_ft = max(elevations_ft)
                min_elev_ft = min(elevations_ft)
                elev_range_ft = max_elev_ft - min_elev_ft
            else:
                max_elev_ft = 0.0
                min_elev_ft = 0.0
                elev_range_ft = 0.0

            lat_avg = sum(p[0] for p in pts) / len(pts)
            lon_avg = sum(p[1] for p in pts) / len(pts)

            # Loop detection
            if len(pts) >= 2:
                first_pt = (pts[0][0], pts[0][1])
                last_pt  = (pts[-1][0], pts[-1][1])
                gap_miles = haversine_miles(first_pt, last_pt)
                route_type = "loop" if gap_miles <= 0.10 else "out-and-back"
                start_point = first_pt
            else:
                route_type = "unknown"
                start_point = (lat_avg, lon_avg)

            raw_name = getattr(gpx, "name", None) or path.stem
            name = normalize_name(str(raw_name))
            rid = stable_route_id(path, pts)

            miles = max(0.1, dist)
            gain_per_mile = gain_ft / miles

            # =========================================================
            # ENRICHED METADATA (computed from GPX data)
            # =========================================================

            difficulty = _infer_difficulty(
                gain_ft=gain_ft,
                distance_miles=dist,
                max_grade_pct=max_grade_pct,
                avg_grade_pct=avg_grade_pct,
                elev_range_ft=elev_range_ft,
            )

            technicality = _infer_technicality(
                name=name,
                max_grade_pct=max_grade_pct,
                avg_grade_pct=avg_grade_pct,
                grade_variance=grade_variance,
            )

            surface_type = _infer_surface(
                name=name,
                avg_grade_pct=avg_grade_pct,
                gain_per_mile=gain_per_mile,
            )

            scenic_likelihood = _infer_scenic_likelihood(
                name=name,
                gain_per_mile=gain_per_mile,
                elev_range_ft=elev_range_ft,
                max_elev_ft=max_elev_ft,
                route_type=route_type,
            )

# =========================================================
            # OSM-DERIVED ENRICHMENT (Santa Clara County layers)
            # =========================================================
            track_latlng = [(p[0], p[1]) for p in pts]
            osm_data = osm_layers.enrich_route(track_latlng)

            routes.append({
                # route_id is guaranteed stable + unique
                "route_id":    rid,
                "name":        name,
                "location":    "",
                "distance_miles": round(dist, 2),
                "elevation_gain": int(round(gain_ft)),
                "surface_type":   surface_type,       # was "unknown"
                "shade_pct":      0,                   # still 0 — needs external data (Tier 3)
                "scenic_likelihood": round(scenic_likelihood, 3),  # was 0
                "proximity_miles": 0,
                "route_type":     route_type,
                "popularity":     0,                   # still 0 — needs sidecar data (Tier 2c)
                "difficulty":     difficulty,           # was "unknown"
                "technicality":   technicality,         # was "unknown"
                # Computed gradient profile
               "max_grade_pct":  round(max_grade_pct, 1),
                "avg_grade_pct":  round(avg_grade_pct, 1),
                # OSM-derived fields (ground truth from OpenStreetMap)
                # OSM-derived fields (ground truth from OpenStreetMap, Santa Clara County)
                "osm_surface":               osm_data.get("osm_surface", "unknown"),
                "osm_highway":               osm_data.get("osm_highway", "unknown"),
                "osm_smoothness":            osm_data.get("osm_smoothness", ""),
                "osm_bicycle_legal":         osm_data.get("osm_bicycle_legal", True),
                "osm_bicycle_no_pct":        osm_data.get("osm_bicycle_no_pct", 0.0),
                "osm_bicycle_no_max_run_m":  osm_data.get("osm_bicycle_no_max_run_m", 0.0),
                "osm_horse_legal":           osm_data.get("osm_horse_legal", False),
                "osm_dog_allowed":           osm_data.get("osm_dog_allowed", None),
                "osm_technicality":          osm_data.get("osm_technicality", 0),
                "osm_sac_scale":             osm_data.get("osm_sac_scale", 0),
                "osm_mtb_scale":             osm_data.get("osm_mtb_scale", 0),
                "osm_has_trailhead_parking": osm_data.get("osm_has_trailhead_parking", False),
                "osm_free_parking":          osm_data.get("osm_free_parking", False),
                "osm_scenic_poi_count":      osm_data.get("osm_scenic_poi_count", 0),
                "osm_water_count":           osm_data.get("osm_water_count", 0),
                "osm_drinking_water_count":  osm_data.get("osm_drinking_water_count", 0),
                "osm_restroom_count":        osm_data.get("osm_restroom_count", 0),
                "osm_shade_pct":             osm_data.get("osm_shade_pct", 0),
                "osm_park_name":             osm_data.get("osm_park_name", ""),
                "osm_park_operator":         osm_data.get("osm_park_operator", ""),
                "osm_park_dog_policy":       osm_data.get("osm_park_dog_policy", ""),
                "osm_park_fee":              osm_data.get("osm_park_fee", ""),
                "osm_park_assignment":       osm_data.get("osm_park_assignment", ""),
                "osm_park_overlap_pct":      osm_data.get("osm_park_overlap_pct", 0.0),
                "osm_picnic_count":          osm_data.get("osm_picnic_count", 0),
                "osm_camping_count":         osm_data.get("osm_camping_count", 0),
                # Internal-only fields
                "_centroid":    (lat_avg, lon_avg),
                "_start_point": start_point,
                "_path":        str(path),
            })

        except Exception as e:
            _record(rejected, path.name, reason="parse_error", detail=str(e))
            print(f"[GPX] REJECTED {path.name}: parse_error ({e})")

    reasons: Dict[str, int] = {}
    for item in rejected:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    LAST_LOAD_REPORT.clear()
    LAST_LOAD_REPORT.update({
        "scanned": scanned,
        "loaded": len(routes),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "rejected_by_reason": reasons,
        "flagged": flagged,
    })
    print(f"[GPX] Validation: scanned={scanned} loaded={len(routes)} rejected={len(rejected)} {reasons} flagged={len(flagged)}")

    # =========================================================
    # Post-load: log enrichment summary
    # =========================================================
    if routes:
        diff_counts = {}
        surf_counts = {}
        tech_counts = {}
        for r in routes:
            diff_counts[r["difficulty"]] = diff_counts.get(r["difficulty"], 0) + 1
            surf_counts[r["surface_type"]] = surf_counts.get(r["surface_type"], 0) + 1
            tech_counts[r["technicality"]] = tech_counts.get(r["technicality"], 0) + 1
        scenic_vals = [r["scenic_likelihood"] for r in routes]
        avg_scenic = sum(scenic_vals) / len(scenic_vals)
        print(f"[GPX] Enrichment summary:")
        print(f"  Difficulty:   {diff_counts}")
        print(f"  Surface:      {surf_counts}")
        print(f"  Technicality: {tech_counts}")
        print(f"  Scenic avg:   {avg_scenic:.3f}  min={min(scenic_vals):.3f}  max={max(scenic_vals):.3f}")

# OSM summary
        osm_surf_counts = {}
        osm_highway_counts = {}
        osm_park_counts = {}
        osm_parking_yes = 0
        osm_scenic_total = 0
        osm_water_total = 0
        osm_drinking_total = 0
        osm_restroom_total = 0
        osm_shade_vals = []
        osm_routes_in_park = 0
        for r in routes:
            osm_surf_counts[r.get("osm_surface", "unknown")] = osm_surf_counts.get(r.get("osm_surface", "unknown"), 0) + 1
            osm_highway_counts[r.get("osm_highway", "unknown")] = osm_highway_counts.get(r.get("osm_highway", "unknown"), 0) + 1
            park = r.get("osm_park_name", "") or "(none)"
            osm_park_counts[park] = osm_park_counts.get(park, 0) + 1
            if r.get("osm_has_trailhead_parking"):
                osm_parking_yes += 1
            if r.get("osm_park_name"):
                osm_routes_in_park += 1
            osm_scenic_total += r.get("osm_scenic_poi_count", 0)
            osm_water_total += r.get("osm_water_count", 0)
            osm_drinking_total += r.get("osm_drinking_water_count", 0)
            osm_restroom_total += r.get("osm_restroom_count", 0)
            osm_shade_vals.append(r.get("osm_shade_pct", 0))
        avg_shade = sum(osm_shade_vals) / len(osm_shade_vals) if osm_shade_vals else 0
        print(f"[OSM] Surface:        {osm_surf_counts}")
        print(f"[OSM] Highway type:   {osm_highway_counts}")
        print(f"[OSM] Trailhead parking: {osm_parking_yes}/{len(routes)} routes")
        print(f"[OSM] Within named park: {osm_routes_in_park}/{len(routes)} routes")
        print(f"[OSM] Avg shade:      {avg_shade:.1f}%")
        print(f"[OSM] Water features: {osm_water_total} ({osm_drinking_total} drinking)")
        print(f"[OSM] Restrooms near routes: {osm_restroom_total}")
        print(f"[OSM] Scenic POIs:    {osm_scenic_total}")
        # Top 5 parks by route count
        top_parks = sorted(osm_park_counts.items(), key=lambda x: -x[1])[:5]
        print(f"[OSM] Top parks:      {dict(top_parks)}")
    return routes
