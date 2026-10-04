import csv
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sync_supabase as ss


def _engine():
    return [
        {"route_id": "abc123", "name": "Bob's Loop", "_start_point": (37.0, -121.0), "distance_miles": 3.2,
         "elevation_gain": 410, "osm_bicycle_legal": False, "osm_dog_allowed": None, "osm_park_name": "Test Park"},
        {"route_id": "def456", "name": "Ridge", "_start_point": (37.1, -121.1), "distance_miles": 5.0,
         "elevation_gain": 900, "osm_bicycle_legal": True, "osm_dog_allowed": True, "osm_park_name": ""},
    ]


def _sb():
    def row(i, n, lat, lng, d, g):
        return {"id": i, "name": n, "start": (lat, lng), "distance": d, "elevation_gain": g}
    return [
        row("id-1", "Bob&#39;s Loop", 37.0001, -121.0001, 3.0, 300),
        row("id-2", "Ridge", 37.1, -121.1, 5.1, 950),
        row("id-3", "way/421156644", 37.5, -121.5, 0.0, 0),
    ]


def test_generator_offline_and_correct(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    rows, fixes, junk, _ = ss.build(_engine(), _sb())
    sql = ss.build_sql(rows, fixes, junk)
    ss.write_outputs(rows, sql, tmp_path)
    assert [r["supabase_id"] for r in rows] == ["id-1", "id-2"]
    assert sql.index("SELECT") < sql.index("BEGIN;") < sql.index("COMMIT;")
    assert "ADD COLUMN IF NOT EXISTS engine_route_id text" in sql
    assert "WHERE engine_route_id IS NOT NULL" in sql
    assert "SET name = 'Bob''s Loop' WHERE id = 'id-1'" in sql
    assert "-- DELETE FROM routes WHERE id = 'id-3'" in sql
    assert "\nDELETE" not in sql
    assert "bikes_allowed = false" in sql and "dogs_allowed" in sql.split("id-1")[-1]
    with open(tmp_path / "supabase_sync.csv") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_far_start_marked_review():
    sb = _sb()
    sb[0]["start"] = (38.0, -121.0)
    rows, _, _, _ = ss.build(_engine(), sb)
    assert rows[0]["status"] == "review"
    assert "-- REVIEW UPDATE" in ss.build_sql(rows, [], [])
