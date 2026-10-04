import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_drift as cd


def test_repo_has_no_unexpected_drift(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", boom)
    assert cd.main([]) == 0, capsys.readouterr().out


def test_compare_detects_drift():
    eng = [{"route_id": "e1", "name": "A"}, {"route_id": "e2", "name": "New"}, {"route_id": "e3", "name": "Other"}]
    snap = [
        {"id": "s1", "name": "A", "engine_route_id": "e1"},
        {"id": "s2", "name": "Orphan", "engine_route_id": ""},
        {"id": "s3", "name": "Allowed", "engine_route_id": ""},
        {"id": "s4", "name": "Oth&#39;er", "engine_route_id": "e3"},
    ]
    rep = cd.compare(eng, snap, {"s3", "gone"})
    assert rep["engine_missing_from_snapshot"] == ["New [e2]"]
    assert rep["snapshot_not_in_engine"] == ["Orphan [s2]"]
    assert len(rep["name_mismatch"]) == 1 and "e3" in rep["name_mismatch"][0]
    assert rep["stale_allowlist"] == ["gone"]
