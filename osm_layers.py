"""
OSM data layer loader and query interface.

Loads GeoJSON files at startup (lazy on first use) and builds R-tree spatial
indexes for fast point-in-region and nearest-neighbor queries.

All layers cover Santa Clara County, CA.

Derivation notes (Phase 2 data fixes)
- osm_bicycle_legal: sampled points are snapped to the nearest line way within 25 m.
  osm_bicycle_no_pct is the share of snapped samples on ways tagged bicycle=no.
  The route is bike-legal when that share is <= BICYCLE_NO_MAX_FRACTION. Untagged ways
  are unknown, not prohibited. Routes with no snapped way default to legal.
- osm_park_name: park containing the route start point (osm_park_assignment="start_point");
  otherwise the named protected area containing at least PARK_FOOTPRINT_MIN_FRACTION of
  the track length ("footprint"); otherwise "" with osm_park_assignment="unassigned".
  osm_park_overlap_pct is the share of track length inside the assigned (or best
  candidate) park.
"""
import json
import math
from pathlib import Path
from typing import Optional
from shapely import affinity
from shapely.geometry import shape, Point, LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

DATA_DIR = Path(__file__).parent / "data" / "osm"


class OSMLayer:
    """Holds a spatial index + properties for one GeoJSON layer."""

    def __init__(self, name: str, filename: str):
        self.name = name
        self.path = DATA_DIR / filename
        self.geometries: list = []
        self.properties: list[dict] = []
        self.index: Optional[STRtree] = None
        self._linear_idx: Optional[set] = None

    def load(self) -> None:
        if not self.path.exists():
            print(f"[osm] WARNING: {self.path} not found, layer '{self.name}' disabled", flush=True)
            return

        try:
            with open(self.path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[osm] ERROR opening {self.path}: {type(e).__name__}: {e}", flush=True)
            return

        skipped = 0
        for feat in data.get("features", []):
            try:
                geom = shape(feat["geometry"])
                self.geometries.append(geom)
                self.properties.append(feat.get("properties", {}))
            except Exception:
                skipped += 1
                continue

        try:
            if self.geometries:
                self.index = STRtree(self.geometries)
        except Exception as e:
            print(f"[osm] ERROR building index for '{self.name}': {type(e).__name__}: {e}", flush=True)
            return

        print(f"[osm] Loaded {len(self.geometries)} features from '{self.name}' (skipped {skipped})", flush=True)

    def features_near(self, lat: float, lng: float, radius_m: float = 100) -> list[dict]:
        """Return properties of features whose geometry is within radius_m of (lat,lng)."""
        if self.index is None:
            self.load()
        if self.index is None:
            return []
        radius_deg = radius_m / 111_000
        pt = Point(lng, lat)
        candidates = self.index.query(pt.buffer(radius_deg))
        hits = []
        for c in candidates:
            if isinstance(c, (int,)) or hasattr(c, "item"):
                idx = int(c)
                geom = self.geometries[idx]
            else:
                geom = c
                idx = self.geometries.index(c)
            if geom.distance(pt) <= radius_deg:
                hits.append(self.properties[idx])
        return hits

    def nearest_linear(self, lat: float, lng: float, radius_m: float = 25) -> Optional[dict]:
        """Properties of the closest line feature (not point or polygon) within radius_m."""
        if self.index is None:
            self.load()
        if self.index is None:
            return None
        if self._linear_idx is None:
            self._linear_idx = {
                i for i, g in enumerate(self.geometries)
                if g.geom_type in ("LineString", "MultiLineString")
            }
        radius_deg = radius_m / 111_000
        pt = Point(lng, lat)
        best_idx, best_d = None, None
        for c in self.index.query(pt.buffer(radius_deg)):
            idx = int(c)
            if idx not in self._linear_idx:
                continue
            d = self.geometries[idx].distance(pt)
            if d <= radius_deg and (best_d is None or d < best_d):
                best_idx, best_d = idx, d
        return self.properties[best_idx] if best_idx is not None else None

    def features_containing(self, lat: float, lng: float) -> list[dict]:
        """Return properties of polygon features that contain (lat,lng)."""
        if self.index is None:
            self.load()
        if self.index is None:
            return []
        pt = Point(lng, lat)
        candidates = self.index.query(pt)
        hits = []
        for c in candidates:
            if isinstance(c, (int,)) or hasattr(c, "item"):
                idx = int(c)
                geom = self.geometries[idx]
            else:
                geom = c
                idx = self.geometries.index(c)
            if geom.contains(pt):
                hits.append(self.properties[idx])
        return hits


# Singleton layers — lazy-load on first query
TRAILS         = OSMLayer("trails",          "trails.geojson")
PARKING        = OSMLayer("parking",         "parking.geojson")
SCENIC         = OSMLayer("scenic",          "scenic.geojson")
WATER          = OSMLayer("water",           "water.geojson")
RESTROOMS      = OSMLayer("restrooms",       "restrooms.geojson")
LANDCOVER      = OSMLayer("landcover",       "landcover.geojson")
PROTECTED      = OSMLayer("protected",       "protected.geojson")
PICNIC_CAMPING = OSMLayer("picnic_camping",  "picnic_camping.geojson")
NATURAL_FEAT   = OSMLayer("natural_feat",    "natural_features.geojson")

def load_all() -> None:
    """Eager-load all layers (call at FastAPI startup if desired)."""
    for layer in (TRAILS, PARKING, SCENIC, WATER, RESTROOMS, LANDCOVER, PROTECTED,
                  PICNIC_CAMPING, NATURAL_FEAT):
        if layer.index is None:
            layer.load()


SAC_SCALE_MAP = {
    "hiking": 1, "mountain_hiking": 2, "demanding_mountain_hiking": 3,
    "alpine_hiking": 4, "demanding_alpine_hiking": 5,
    "difficult_alpine_hiking": 6,
}

MTB_SCALE_MAP = {
    "0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 6,
}


PARK_FOOTPRINT_MIN_FRACTION = 0.5
BICYCLE_NO_MAX_FRACTION = 0.10
_LNG_SCALE = math.cos(math.radians(37.2))
_PARK_GROUPS: Optional[dict] = None


def _scale_geom(geom):
    return affinity.scale(geom, xfact=_LNG_SCALE, yfact=1.0, origin=(0, 0))


def _park_groups() -> dict:
    """Named protected-area polygons grouped by name: {name: {"geom", "props"}} (lng scaled to metric-ish)."""
    global _PARK_GROUPS
    if _PARK_GROUPS is not None:
        return _PARK_GROUPS
    if PROTECTED.index is None:
        PROTECTED.load()
    members: dict[str, list] = {}
    for geom, props in zip(PROTECTED.geometries, PROTECTED.properties):
        name = props.get("name")
        if not name or geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        members.setdefault(name, []).append((geom, props))
    groups = {}
    for name, items in members.items():
        scaled = [_scale_geom(g) for g, _ in items]
        preferred = sorted(
            (p for _, p in items),
            key=lambda p: (p.get("boundary") != "protected_area", not p.get("operator")),
        )[0]
        groups[name] = {"geom": unary_union(scaled), "props": preferred}
    _PARK_GROUPS = groups
    return groups


def assign_park_by_footprint(track_points: list[tuple[float, float]]) -> tuple[str, float, dict]:
    """
    Majority-of-track overlap against named protected areas.
    Returns (best_name, fraction_of_track_length_inside, properties). Empty name if none overlaps.
    """
    if len(track_points) < 2:
        return "", 0.0, {}
    line = LineString([(lng * _LNG_SCALE, lat) for lat, lng in track_points])
    total = line.length
    if total <= 0:
        return "", 0.0, {}
    best_name, best_frac, best_props = "", 0.0, {}
    for name, grp in _park_groups().items():
        geom = grp["geom"]
        if not geom.intersects(line):
            continue
        frac = line.intersection(geom).length / total
        if frac > best_frac:
            best_name, best_frac, best_props = name, frac, grp["props"]
    return best_name, best_frac, best_props


def enrich_route(track_points: list[tuple[float, float]]) -> dict:
    """
    Given a list of (lat, lng) points from a GPX track, return aggregate
    OSM-derived features for the route.
    """
    if not track_points:
        return {}
        
    # Sample at most 50 points along the route for tag inference
    step = max(1, len(track_points) // 50)
    sampled = track_points[::step]

    # --- Trail tag aggregation ---
    surface_counts: dict[str, int] = {}
    highway_counts: dict[str, int] = {}
    smoothness_counts: dict[str, int] = {}
    bicycle_no_samples = 0
    bicycle_way_samples = 0
    horse_legal = False
    dog_allowed = None  # None = unknown; True/False if explicitly tagged
    sac_samples = []
    mtb_samples = []
    incline_samples = []
    lit_count = 0

    for lat, lng in sampled:
        hits = TRAILS.features_near(lat, lng, radius_m=25)
        if not hits:
            continue
        tags = hits[0]
        if surface := tags.get("surface"):
            surface_counts[surface] = surface_counts.get(surface, 0) + 1
        if highway := tags.get("highway"):
            highway_counts[highway] = highway_counts.get(highway, 0) + 1
        if smooth := tags.get("smoothness"):
            smoothness_counts[smooth] = smoothness_counts.get(smooth, 0) + 1
        way = TRAILS.nearest_linear(lat, lng, radius_m=25)
        if way is not None:
            bicycle_way_samples += 1
            if way.get("bicycle") == "no":
                bicycle_no_samples += 1
        if tags.get("horse") in ("yes", "designated"):
            horse_legal = True
        if (dog_tag := tags.get("dog")) is not None:
            dog_allowed = dog_tag in ("yes", "leashed")
        if sac := tags.get("sac_scale"):
            sac_samples.append(SAC_SCALE_MAP.get(sac, 0))
        if mtb := tags.get("mtb:scale"):
            mtb_samples.append(MTB_SCALE_MAP.get(str(mtb), 0))
        if tags.get("lit") == "yes":
            lit_count += 1

    # --- Trailhead parking (within 300m of route start) ---
    start_lat, start_lng = track_points[0]
    nearby_parking = PARKING.features_near(start_lat, start_lng, radius_m=300)
    has_trailhead_parking = len(nearby_parking) > 0
    free_parking = any(p.get("fee") in (None, "no") for p in nearby_parking)

    # --- Scenic POIs along route (deduped by name+type) ---
    # Combines core scenic (peaks/viewpoints/waterfalls) with natural features
    # (cliffs/rocks/caves/arches) for richer POI scoring.
    scenic_count = 0
    seen_scenic = set()
    for lat, lng in sampled:
        for h in SCENIC.features_near(lat, lng, radius_m=150):
            key = (h.get("name", ""), h.get("natural", h.get("tourism", "")))
            if key not in seen_scenic:
                seen_scenic.add(key)
                scenic_count += 1
        for h in NATURAL_FEAT.features_near(lat, lng, radius_m=100):
            key = (h.get("name", ""), h.get("natural", ""))
            if key not in seen_scenic:
                seen_scenic.add(key)
                scenic_count += 1

    # --- Picnic & camping facilities along route ---
    picnic_count = 0
    camping_count = 0
    seen_picnic = set()
    for lat, lng in sampled:
        for h in PICNIC_CAMPING.features_near(lat, lng, radius_m=150):
            key = (h.get("name", "") or id(h),
                   h.get("tourism", h.get("leisure", "")))
            if key in seen_picnic:
                continue
            seen_picnic.add(key)
            tag = h.get("tourism") or h.get("leisure") or ""
            if "camp" in tag:
                camping_count += 1
            else:
                picnic_count += 1

    # --- Water access along route (deduped) ---
    water_count = 0
    drinking_water_count = 0
    seen_water = set()
    for lat, lng in sampled:
        for h in WATER.features_near(lat, lng, radius_m=100):
            key = (h.get("name", ""), h.get("waterway", h.get("amenity", h.get("natural", ""))))
            if key in seen_water:
                continue
            seen_water.add(key)
            water_count += 1
            if h.get("amenity") == "drinking_water":
                drinking_water_count += 1

    # --- Restrooms within 200m of route ---
    restroom_count = 0
    seen_restroom = set()
    for lat, lng in sampled:
        for h in RESTROOMS.features_near(lat, lng, radius_m=200):
            key = h.get("name", "") or id(h)
            if key not in seen_restroom:
                seen_restroom.add(key)
                restroom_count += 1
                
   # --- Protected area / land manager (use route start as the reference point) ---
    protected_areas = PROTECTED.features_containing(start_lat, start_lng)
    park_name = ""
    park_operator = ""
    park_dog_policy = ""
    park_fee = ""
    park_assignment = "unassigned"
    if protected_areas:
        # Prefer features that have a name
        named = [p for p in protected_areas if p.get("name")]
        chosen = named[0] if named else protected_areas[0]
        park_name = chosen.get("name", "")
        park_operator = chosen.get("operator", "")
        park_dog_policy = chosen.get("dog", "")
        park_fee = chosen.get("fee", "")
        park_assignment = "start_point"

    fp_name, fp_fraction, fp_props = assign_park_by_footprint(track_points)
    if park_name:
        group = _park_groups().get(park_name)
        if group is not None and fp_name == park_name:
            park_overlap = fp_fraction
        elif group is not None:
            line = LineString([(lng * _LNG_SCALE, lat) for lat, lng in track_points])
            park_overlap = line.intersection(group["geom"]).length / line.length if line.length > 0 else 0.0
        else:
            park_overlap = 0.0
    elif fp_name and fp_fraction >= PARK_FOOTPRINT_MIN_FRACTION:
        park_name = fp_name
        park_operator = fp_props.get("operator", "")
        park_dog_policy = fp_props.get("dog", "")
        park_fee = fp_props.get("fee", "")
        park_assignment = "footprint"
        park_overlap = fp_fraction
    else:
        park_overlap = fp_fraction

    # --- Shade estimate (% of sampled points strictly inside forest/wood polygons) ---
    # OSM landcover coverage varies by park. Strict containment works well in
    # well-mapped parks (Sanborn, Uvas, Villa Montalvo). For parks with known
    # tree cover but missing OSM landcover polygons, we apply a fallback below.
    # Future: replace with NDVI from Sentinel-2 imagery for ground-truth.
    shade_hits = 0
    for lat, lng in sampled:
        in_forest = LANDCOVER.features_containing(lat, lng)
        if any(f.get("natural") == "wood" or f.get("landuse") == "forest" for f in in_forest):
            shade_hits += 1
    shade_pct = round(100 * shade_hits / len(sampled)) if sampled else 0

    # OSM coverage gap fallback: parks with known dense tree cover but
    # under-tagged landcover get a defensible default. Only applies when
    # OSM-derived shade is implausibly low (<20%).
    OSM_SHADE_FALLBACKS = {
        "Mount Madonna County Park": 80,  # Dense redwood/tan oak; OSM under-tagged
    }
    if shade_pct < 20 and park_name in OSM_SHADE_FALLBACKS:
        shade_pct = OSM_SHADE_FALLBACKS[park_name]

    # --- Summarize ---
    dominant_surface = max(surface_counts, key=surface_counts.get) if surface_counts else "unknown"
    dominant_highway = max(highway_counts, key=highway_counts.get) if highway_counts else "unknown"
    dominant_smoothness = max(smoothness_counts, key=smoothness_counts.get) if smoothness_counts else ""
    bicycle_no_pct = (
        round(100 * bicycle_no_samples / bicycle_way_samples, 1) if bicycle_way_samples else 0.0
    )
    bicycle_legal = (bicycle_no_samples / bicycle_way_samples) <= BICYCLE_NO_MAX_FRACTION \
        if bicycle_way_samples else True
    avg_sac = round(sum(sac_samples) / len(sac_samples), 2) if sac_samples else 0
    avg_mtb = round(sum(mtb_samples) / len(mtb_samples), 2) if mtb_samples else 0
    # Combined technicality: prefer mtb scale if present, else sac scale
    osm_technicality = avg_mtb if avg_mtb > 0 else avg_sac

    return {
        "osm_surface":               dominant_surface,
        "osm_highway":               dominant_highway,
        "osm_smoothness":            dominant_smoothness,
        "osm_bicycle_legal":         bicycle_legal,
        "osm_bicycle_no_pct":        bicycle_no_pct,
        "osm_horse_legal":           horse_legal,
        "osm_dog_allowed":           dog_allowed,
        "osm_technicality":          osm_technicality,
        "osm_sac_scale":             avg_sac,
        "osm_mtb_scale":             avg_mtb,
        "osm_has_trailhead_parking": has_trailhead_parking,
        "osm_free_parking":          free_parking,
        "osm_scenic_poi_count":      scenic_count,
        "osm_water_count":           water_count,
        "osm_drinking_water_count":  drinking_water_count,
        "osm_restroom_count":        restroom_count,
        "osm_shade_pct":             shade_pct,
        "osm_park_name":             park_name,
        "osm_park_operator":         park_operator,
        "osm_park_dog_policy":       park_dog_policy,
        "osm_park_fee":              park_fee,
        "osm_park_assignment":       park_assignment,
        "osm_park_overlap_pct":      round(100 * park_overlap, 1),
        "osm_picnic_count":          picnic_count,
        "osm_camping_count":         camping_count,
    }
