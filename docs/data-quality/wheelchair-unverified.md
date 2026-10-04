# Wheelchair: unverified routes

## Rule

The engine returns a route for a wheelchair-gated query only when `wheelchair_access(route) == "verified"` (`engine.py`). That requires both:

- GPX heuristic `surface_type == "paved"`, and
- `osm_surface` in {paved, asphalt, concrete, paving_stones}.

Anything else is "unverified" and fails closed. The 10 routes below are heuristic-paved, but OSM has no surface data (`osm_surface == "unknown"`), so the paved claim cannot be confirmed. They are excluded deliberately, since a false positive harms a wheelchair user. The benchmark ground truth (`tests/benchmark/harness.py`) mirrors this rule, and `tests/test_wheelchair_unverified.py` guards these routes.

## Counts (255 routes)

| State | Count |
|---|---|
| Verified (GPX paved and OSM paved) | 26 |
| Unverified, GPX and OSM disagree | 50 |
| Unverified, OSM unknown (listed below) | 10 |
| Not accessible (neither says paved) | 169 |

## The 10 routes

| route_id | name | park | distance_mi | surface_type | osm_surface | gpx_file |
|---|---|---|---|---|---|---|
| 270bf04821300c5d | Calaveras Loop | Coyote Lake - Harvey Bear Ranch County Park | 7.82 | paved | unknown | Calaveras Loop.gpx |
| 423f74902dd28a48 | Rosendin Pond Loop | Anderson Lake County Park | 0.99 | paved | unknown | Rosendin Pond Loop.gpx |
| 81986ce0261a6b86 | Eagle Lake Out & Back | Joseph D. Grant County Park | 6.32 | paved | unknown | Eagle Lake Out & Back.gpx |
| 8fd04735ae9e86c4 | Two Moon Lake | Long Ridge Open Space Preserve | 1.01 | paved | unknown | Two Moon Lake.gpx |
| a20b399f209bf9f5 | Harvey Bear Ranch | Coyote Lake - Harvey Bear Ranch County Park | 2.03 | paved | unknown | Harvey Bear Ranch.gpx |
| b44257a6bcebb7e8 | Spring Valley Pond Loop | Ed R. Levin County Park | 0.41 | paved | unknown | Spring Valley Pond Loop.gpx |
| c1d1aad6a8f1bc69 | Savannah Loop via Mendoza | Coyote Lake - Harvey Bear Ranch County Park | 11.03 | paved | unknown | Savannah Loop via Mendoza.gpx |
| ce6e59b9f1d6bda3 | Cottonwood Lake Lap | Hellyer County Park | 0.57 | paved | unknown | Cottonwood Lake Lap.gpx |
| d9dfd6188a37265e | Brush Loop | Joseph D. Grant County Park | 4.75 | paved | unknown | Brush Loop.gpx |
| f2cf98fef0307774 | Hellyer Lap | Hellyer County Park | 0.87 | paved | unknown | Hellyer Lap.gpx |

## Resolution

Per route, either:

1. Field check: confirm the surface on site or from imagery, then add `surface=paved` (or `asphalt`/`concrete`) to the way in OpenStreetMap, or
2. If the route is not actually paved, correct the GPX `surface_type` so the heuristic no longer says paved.

After an OSM edit, re-run the enrichment pipeline so `osm_surface` updates; the route then becomes verified automatically.

## Regenerate

    MAPBOX_TOKEN= python scripts/list_wheelchair_unverified.py
