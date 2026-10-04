import ast
import dataclasses
import typing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
INTERNAL_LOADER_KEYS = {"_centroid", "_path", "_start_point"}
START_POINT_FLAT = {"start_lat", "start_lng"}
ENGINE_ONLY_FIELDS = {"_start_point", "_centroid", "_path"}


@pytest.fixture(scope="module")
def app_module():
    import app
    return app


@pytest.fixture(scope="module")
def route_in(app_module):
    return app_module.RouteIn


@pytest.fixture(scope="module")
def raw_routes(app_module):
    assert app_module._RAW_GPX_ROUTES, "gpx_loader returned no routes from ./data/gpx"
    return app_module._RAW_GPX_ROUTES


def _return_dict_keys(source_file: str, func_name: str, anchor_call: str = None):
    tree = ast.parse((ROOT / source_file).read_text(encoding="utf-8"))
    keys = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == func_name:
            for node in ast.walk(fn):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                if anchor_call and isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Attribute) and f.attr == "append" and node.args \
                            and isinstance(node.args[0], ast.Dict):
                        keys |= {k.value for k in node.args[0].keys if isinstance(k, ast.Constant)}
    return keys


def _model_fields(route_in):
    return route_in.model_fields


def _required(route_in):
    return {n for n, f in _model_fields(route_in).items() if f.is_required()}


def test_routein_covers_static_enrich_route_keys(route_in):
    keys = _return_dict_keys("osm_layers.py", "enrich_route")
    assert keys, "could not parse enrich_route return dict"
    missing = keys - set(_model_fields(route_in))
    assert not missing, f"enrich_route keys missing from RouteIn (would be dropped): {sorted(missing)}"


def test_routein_covers_static_gpx_loader_keys(route_in):
    keys = _return_dict_keys("gpx_loader.py", "load_routes_from_gpx_dir", anchor_call="append")
    assert keys, "could not parse gpx_loader routes.append dict"
    public = keys - INTERNAL_LOADER_KEYS
    missing = public - set(_model_fields(route_in))
    assert not missing, f"gpx_loader keys missing from RouteIn (would be dropped): {sorted(missing)}"
    if "_start_point" in keys:
        assert START_POINT_FLAT <= set(_model_fields(route_in))


def test_routein_covers_runtime_loader_output(route_in, raw_routes):
    seen = set()
    for r in raw_routes:
        seen |= set(r)
    public = seen - INTERNAL_LOADER_KEYS
    missing = public - set(_model_fields(route_in))
    assert not missing, f"runtime loader keys missing from RouteIn: {sorted(missing)}"


def test_routein_covers_runtime_enrich_route_output(route_in, raw_routes):
    import gpxpy
    import osm_layers
    path = Path(raw_routes[0]["_path"])
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        gpx = gpxpy.parse(f)
    pts = [(p.latitude, p.longitude) for t in gpx.tracks for s in t.segments for p in s.points]
    assert len(pts) >= 2
    out = osm_layers.enrich_route(pts)
    assert out, "enrich_route returned empty for a real track"
    missing = set(out) - set(_model_fields(route_in))
    assert not missing, f"enrich_route runtime keys missing from RouteIn: {sorted(missing)}"


def test_every_loader_route_roundtrips_without_dropped_or_changed_fields(route_in, raw_routes, app_module):
    for raw in raw_routes:
        flat = app_module._strip_internal(raw)
        dumped = route_in(**flat).model_dump()
        dropped = set(flat) - set(dumped)
        assert not dropped, f"{raw['name']}: dropped {sorted(dropped)}"
        for k, v in flat.items():
            assert dumped[k] == v or (isinstance(v, (int, float)) and float(dumped[k]) == float(v)), \
                f"{raw['name']}: {k} changed {v!r} -> {dumped[k]!r}"


def test_loader_value_types_match_routein_annotations(route_in, raw_routes, app_module):
    def base_types(ann):
        origin = typing.get_origin(ann)
        if origin is typing.Union:
            return tuple(t for a in typing.get_args(ann) for t in base_types(a))
        return (ann,)

    bad = []
    for raw in raw_routes:
        flat = app_module._strip_internal(raw)
        for k, v in flat.items():
            ann = _model_fields(route_in)[k].annotation
            allowed = base_types(ann)
            if v is None:
                ok = type(None) in allowed
            elif float in allowed:
                ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            elif int in allowed:
                ok = isinstance(v, int) and not isinstance(v, bool)
            else:
                ok = isinstance(v, allowed)
            if not ok:
                bad.append((raw["name"], k, type(v).__name__, str(ann)))
    assert not bad, f"loader value types incompatible with RouteIn: {bad[:10]}"


def test_required_routein_fields_always_emitted_by_loader(route_in, raw_routes, app_module):
    required = _required(route_in)
    for raw in raw_routes:
        flat = app_module._strip_internal(raw)
        missing = required - set(flat)
        assert not missing, f"{raw['name']}: required RouteIn fields absent from loader: {sorted(missing)}"
        for k in required:
            assert flat[k] is not None, f"{raw['name']}: required field {k} is None"


def test_routein_fields_match_engine_route_dataclass(route_in):
    from engine import Route
    engine_fields = {f.name for f in dataclasses.fields(Route)}
    routein_fields = set(_model_fields(route_in))
    expected_engine = (routein_fields - START_POINT_FLAT) | ({"_start_point"} & engine_fields)
    missing_in_routein = (engine_fields - ENGINE_ONLY_FIELDS) - routein_fields
    missing_in_engine = (routein_fields - START_POINT_FLAT) - engine_fields
    assert not missing_in_routein, f"engine.Route fields not in RouteIn (cannot be populated): {sorted(missing_in_routein)}"
    assert not missing_in_engine, f"RouteIn fields not in engine.Route (to_engine_route would fail): {sorted(missing_in_engine)}"
    assert expected_engine <= engine_fields


def test_routein_defaults_match_engine_route_defaults(route_in):
    from engine import Route
    mismatches = []
    for f in dataclasses.fields(Route):
        if f.name not in _model_fields(route_in) or f.name.startswith("_"):
            continue
        if f.default is dataclasses.MISSING:
            continue
        pf = _model_fields(route_in)[f.name]
        if pf.is_required():
            mismatches.append((f.name, "required in RouteIn, default in Route"))
        elif pf.default != f.default:
            mismatches.append((f.name, f"RouteIn={pf.default!r} Route={f.default!r}"))
    assert not mismatches, f"default mismatch: {mismatches}"


def test_to_engine_route_builds_for_every_route(app_module):
    for r in app_module.ROUTE_DB:
        er = r.to_engine_route()
        assert er.route_id == r.route_id
        assert getattr(er, "_start_point", None) is not None or r.start_lat is None
