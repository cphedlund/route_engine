import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SNAPSHOT = ROOT / "data" / "supabase_snapshot.csv"
ALLOWLIST = ROOT / "data" / "supabase_only_allowlist.csv"


def norm(s):
    return " ".join(html.unescape(s or "").split())


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare(engine, snapshot, allowlist_ids):
    snap_by_eid = {r["engine_route_id"]: r for r in snapshot if r.get("engine_route_id")}
    eng_ids = {e["route_id"] for e in engine}
    report = {
        "engine_missing_from_snapshot": sorted(e["name"] + " [" + e["route_id"] + "]" for e in engine if e["route_id"] not in snap_by_eid),
        "snapshot_not_in_engine": sorted(
            r["name"] + " [" + r["id"] + "]" for r in snapshot
            if r["id"] not in allowlist_ids and (not r.get("engine_route_id") or r["engine_route_id"] not in eng_ids)
        ),
        "name_mismatch": sorted(
            f"{e['name']!r} vs {snap_by_eid[e['route_id']]['name']!r} [{e['route_id']}]"
            for e in engine if e["route_id"] in snap_by_eid and norm(e["name"]) != norm(snap_by_eid[e["route_id"]]["name"])
        ),
        "stale_allowlist": sorted(i for i in allowlist_ids if i not in {r["id"] for r in snapshot}),
    }
    return report


def refresh(export_json, out_path):
    rows = json.loads(Path(export_json).read_text(encoding="utf-8"))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "engine_route_id"])
        for r in sorted(rows, key=lambda r: r["id"]):
            w.writerow([r["id"], r["name"], r.get("engine_route_id") or ""])


def main(argv=None, engine=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("--allowlist", default=str(ALLOWLIST))
    ap.add_argument("--gpx-dir", default=str(ROOT / "data" / "gpx"))
    ap.add_argument("--refresh", metavar="EXPORT_JSON", help="rewrite the snapshot from a PostgREST JSON export (id,name,engine_route_id)")
    a = ap.parse_args(argv)
    if a.refresh:
        refresh(a.refresh, a.snapshot)
        print(f"snapshot written: {a.snapshot}")
        return 0
    if engine is None:
        from gpx_loader import load_routes_from_gpx_dir
        engine = load_routes_from_gpx_dir(a.gpx_dir)
    allow = {r["id"] for r in read_csv(a.allowlist)}
    rep = compare(engine, read_csv(a.snapshot), allow)
    bad = 0
    for k, v in rep.items():
        print(f"{k}: {len(v)}")
        for x in v:
            print("  " + x)
        bad += len(v)
    print("DRIFT" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
