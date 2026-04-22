"""
OSM data layer loader and query interface.

Loads GeoJSON files at startup (lazy on first use) and builds R-tree spatial
indexes for fast point-in-region and nearest-neighbor queries.

All layers cover Santa Clara County, CA.
"""
import json
from pathlib import Path
from typing import Optional
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

DATA_DIR = Path(__file__).parent / "data" / "osm"
print(f"[OSM PATH DEBUG] DATA_DIR={DATA_DIR.resolve()}, exists={DATA_DIR.exists()}", flush=True)
if DATA_DIR.exists():
    files = list(DATA_DIR.glob("*.geojson"))
    print(f"[OSM PATH DEBUG] Found {len(files)} geojson files", flush=True)


class OSMLayer:
    """Holds a spatial index + properties for one GeoJSON layer."""

    def __init__(self, name: str, filename: str):
        self.name = name
        self.path = DATA_DIR / filename
        self.geometries: list = []
        self.properties: list[dict] = []
        self.index: Optional[STRtree] = None

    def load(self) -> None:
        if not self.path.exists():
            print(f"[osm] WARNING: {self.path} not found, layer '{self.name}' disabled")
            return

        with open(self.path) as f:
            data = json.load(f)

        for feat in data.get("features", []):
            try:
                geom = shape(feat["geometry"])
                self.geometries.append(geom)
                self.properties.append(feat.get("properties", {}))
            except Exception:
                continue

        if self.geometries:
            self.index = STRtree(self.geometries)
        print(f"[osm] Loaded {len(self.geometries)} features from '{self.name}'")

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


def enrich_route(track_points: list[tuple[float, float]]) -> dict:
    """
    Given a list of (lat, lng) points from a GPX track, return aggregate
    OSM-derived features for the route.
    """
    if not track_points:
        return {}
        
        # DEBUG: confirm enrichment is actually running
    if not hasattr(enrich_route, "_debug_logged"):
        enrich_route._debug_logged = True
        sample = LANDCOVER.features_near(track_points[0][0], track_points[0][1], radius_m=100)
        print(f"[ENRICH DEBUG] First call: track_points={len(track_points)}, "
              f"LANDCOVER.index is None: {LANDCOVER.index is None}, "
              f"sample landcover hits at start: {len(sample)}", flush=True)
        
    # Sample at most 50 points along the route for tag inference
    step = max(1, len(track_points) // 50)
    sampled = track_points[::step]

    # --- Trail tag aggregation ---
    surface_counts: dict[str, int] = {}
    highway_counts: dict[str, int] = {}
    smoothness_counts: dict[str, int] = {}
    bicycle_legal = True
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
        if tags.get("bicycle") == "no":
            bicycle_legal = False
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
    if protected_areas:
        # Prefer features that have a name
        named = [p for p in protected_areas if p.get("name")]
        chosen = named[0] if named else protected_areas[0]
        park_name = chosen.get("name", "")
        park_operator = chosen.get("operator", "")
        park_dog_policy = chosen.get("dog", "")
        park_fee = chosen.get("fee", "")

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
    avg_sac = round(sum(sac_samples) / len(sac_samples), 2) if sac_samples else 0
    avg_mtb = round(sum(mtb_samples) / len(mtb_samples), 2) if mtb_samples else 0
    # Combined technicality: prefer mtb scale if present, else sac scale
    osm_technicality = avg_mtb if avg_mtb > 0 else avg_sac

    return {
        "osm_surface":               dominant_surface,
        "osm_highway":               dominant_highway,
        "osm_smoothness":            dominant_smoothness,
        "osm_bicycle_legal":         bicycle_legal,
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
        "osm_picnic_count":          picnic_count,
        "osm_camping_count":         camping_count,
    }
