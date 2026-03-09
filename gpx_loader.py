import math
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
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
    dl = math.radians(lon2 - lon1)
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
# Loader
# ============================================================

def load_routes_from_gpx_dir(gpx_dir: str) -> List[Dict[str, Any]]:
    base = Path(gpx_dir)
    routes: List[Dict[str, Any]] = []

    for path in sorted(base.rglob("*.gpx")):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                gpx = gpxpy.parse(f)

            pts = _extract_points(gpx)
            if len(pts) < 2:
                continue

            # -------------------------
            # Distance + gradient profile
            # -------------------------
            dist = 0.0
            grades: List[float] = []
            for i in range(1, len(pts)):
                seg_dist_miles = haversine_miles(
                    (pts[i - 1][0], pts[i - 1][1]),
                    (pts[i][0], pts[i][1]),
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

            lat_avg = sum(p[0] for p in pts) / len(pts)
            lon_avg = sum(p[1] for p in pts) / len(pts)

            # Loop detection
            if len(pts) >= 2:
                first_pt = (pts[0][0], pts[0][1])
                last_pt  = (pts[-1][0], pts[-1][1])
                gap_miles = haversine_miles(first_pt, last_pt)
                route_type  = "loop" if gap_miles <= 0.10 else "out-and-back"
                start_point = first_pt
            else:
                route_type  = "unknown"
                start_point = (lat_avg, lon_avg)

            raw_name = getattr(gpx, "name", None) or path.stem
            name = normalize_name(str(raw_name))

            rid = stable_route_id(path, pts)

            routes.append({
                # 🔑 CRITICAL: route_id is now guaranteed stable + unique
                "route_id": rid,

                # Human-readable name (used for UI only)
                "name": name,

                "location": "",
                "distance_miles": round(dist, 2),
                "elevation_gain": int(round(gain_ft)),
                "surface_type": "unknown",
                "shade_pct": 0,
                "scenic_likelihood": 0,
                "proximity_miles": 0,
                "route_type": route_type,
                "popularity": 0,
                "difficulty": "unknown",
                "technicality": "unknown",

                # Computed gradient profile
                "max_grade_pct": round(max_grade_pct, 1),
                "avg_grade_pct": round(avg_grade_pct, 1),

                # Internal-only fields
                "_centroid": (lat_avg, lon_avg),
                "_start_point": start_point,
                "_path": str(path),
            })

        except Exception as e:
            print(f"[GPX] Skipping {path.name}: {e}")

    return routes

