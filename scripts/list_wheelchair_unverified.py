import contextlib
import io
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MAPBOX_TOKEN", "")

with contextlib.redirect_stdout(io.StringIO()):
    import app
    from engine import wheelchair_access



def classify(r):
    class R:
        surface_type = r["surface_type"]
        osm_surface = r["osm_surface"]
    state = wheelchair_access(R)
    if state == "unverified":
        return "unverified_unknown" if r["osm_surface"] == "unknown" and r["surface_type"] == "paved" else "unverified_disagree"
    return state.replace(" ", "_")


def rows():
    out = []
    for raw in app._RAW_GPX_ROUTES:
        raw = dict(raw)
        out.append((raw, classify(raw)))
    return out


def main():
    data = rows()
    counts = Counter(c for _, c in data)
    unknown = sorted(
        (r for r, c in data if c == "unverified_unknown"),
        key=lambda r: r["route_id"],
    )
    print(f"total={len(data)}")
    for k in ("verified", "unverified_disagree", "unverified_unknown", "not_accessible"):
        print(f"{k}={counts.get(k, 0)}")
    print()
    print("| route_id | name | park | distance_mi | surface_type | osm_surface | gpx_file |")
    print("|---|---|---|---|---|---|---|")
    for r in unknown:
        print("| {} | {} | {} | {:.2f} | {} | {} | {} |".format(
            r["route_id"], r["name"].replace("|", "/"), r["osm_park_name"] or "unassigned",
            r["distance_miles"], r["surface_type"], r["osm_surface"], os.path.basename(r["_path"])))


if __name__ == "__main__":
    main()
