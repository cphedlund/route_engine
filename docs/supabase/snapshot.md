# Supabase snapshot and drift check

`data/supabase_snapshot.csv` holds Supabase route ids, names (as stored, HTML entities intact) and the proposed `engine_route_id`. No coordinates or user data. Initial source: `atlas-batch/routes.csv` (Aug 6 export), linked using the matching in `scripts/sync_supabase.py`.

`data/supabase_only_allowlist.csv` lists the 50 known Supabase-only rows (see `docs/data-quality/supabase-only-routes.md`).

Check (offline, also run by `tests/test_drift.py` in CI):

    MAPBOX_TOKEN= python scripts/check_drift.py

Exit 1 on: engine routes absent from the snapshot, snapshot rows neither linked to an engine route nor allowlisted, name mismatches after HTML unescape, or allowlist ids no longer in the snapshot.

Refresh after the sync SQL has been applied (adds the `engine_route_id` column). Read-only GET with the public key; the env file is sourced without printing:

    set -a; source /Users/carsonhedlund/Desktop/atlasnav/.env; set +a
    curl -s --max-time 20 "$VITE_SUPABASE_URL/rest/v1/routes?select=id,name,engine_route_id&limit=1000" -H "apikey: $VITE_SUPABASE_PUBLISHABLE_KEY" -o /tmp/routes_ids.json
    python scripts/check_drift.py --refresh /tmp/routes_ids.json

Then update the allowlist for any new Supabase-only rows and commit both CSVs.
