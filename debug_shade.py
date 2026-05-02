"""Spot-check OSM enrichment on specific routes."""
from gpx_loader import load_routes_from_gpx_dir

routes = load_routes_from_gpx_dir("data/gpx")

# Routes we expect to be very different on shade
TARGETS = {
    "high_shade_expected": [
        "Welch Hurst Trail", "Summit Rock Climb", "Sanborn County Park Loop",
        "Lower Madrone Loop", "Nature Trail", "Tan Oak Loop",
    ],
    "low_shade_expected": [
        "Baylands Full Lap", "Baylands Rollers", "Baylands to 49ers",
        "Trapeze to Bay View", "Bernal Hill Loop", "Vasona Figure 8",
    ],
}

def find(name):
    return next((r for r in routes if r["name"].strip().lower() == name.strip().lower()), None)

print("\n" + "=" * 80)
print("SHADE SPOT-CHECK")
print("=" * 80)

for category, names in TARGETS.items():
    print(f"\n{category.upper()}:")
    for n in names:
        r = find(n)
        if not r:
            print(f"  {n:40s}  [NOT FOUND]")
            continue
        print(f"  {r['name']:40s}  shade={r['osm_shade_pct']:>3}%  "
              f"surface={r['osm_surface']:<12s}  highway={r['osm_highway']:<10s}  "
              f"park={r.get('osm_park_name', '')[:30]}")


# Investigate Tan Oak Loop specifically
print("\n" + "=" * 80)
print("TAN OAK LOOP DEEP DIVE")
print("=" * 80)

import osm_layers
r = find("Tan Oak Loop")
if r:
    # Re-enrich and inspect what landcover is actually near it
    import gpxpy
    with open(r["_path"]) as f:
        gpx = gpxpy.parse(f)
    pts = []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                pts.append((p.latitude, p.longitude))

    print(f"\nTotal points: {len(pts)}")
    print(f"First point: {pts[0]}")
    print(f"Center point: {pts[len(pts)//2]}")

    # Check what landcover features are within 50m of midpoint
    mid_lat, mid_lng = pts[len(pts) // 2]
    nearby = osm_layers.LANDCOVER.features_near(mid_lat, mid_lng, radius_m=50)
    print(f"\nLandcover features within 50m of midpoint: {len(nearby)}")
    for f in nearby[:10]:
        print(f"  {f}")

    containing = osm_layers.LANDCOVER.features_containing(mid_lat, mid_lng)
    print(f"\nPolygons containing midpoint: {len(containing)}")
    for f in containing[:10]:
        print(f"  {f}")
