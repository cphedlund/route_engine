import os
import tempfile
from functools import lru_cache

import fitz
import requests

import atlas_scc
import atlas_mapbox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEO_DIR = os.path.join(BASE_DIR, "data", "park_maps", "atlas-geojson-bundle")
PDF_DIR = os.path.join(BASE_DIR, "data", "park_maps", "SCC Park Maps")
MIN_ON_SHEET = 0.95
MAX_SAMPLE = 300

SHEETS = {
    "almaden-quicksilver_geo.json": "almaden-quicksilver-guide-map.pdf",
    "alviso-marina_geo.json": "Alviso Marina Guide Map.pdf",
    "anderson-lake_geo.json": "anderson-lake-guide-map.pdf",
    "calero_geo.json": "calero-guide-map.pdf",
    "coyote-creek-parkway_geo.json": "coyote-creek-parkway-guide-map.pdf",
    "coyote-lake---harvey-bear-ranch_geo__1_.json": "coyote-lake-harvey-bear-ranch-guide-map.pdf",
    "ed-r_-levin_geo.json": "ed-r-levin-guide-map.pdf",
    "hellyer_geo.json": "Hellyer Guide Map.pdf",
    "joseph-d_-grant_geo.json": "joseph-d-grant-guide-map.pdf",
    "lexington-reservoir_geo.json": "lexington-reservoir-guide-map.pdf",
    "los-gatos-creek_geo.json": "los-gatos-creek-park-guide-map.pdf",
    "martial-cottle_geo.json": "martial-cottle-guide-map.pdf",
    "mount-madonna_geo.json": "mt-madonna-guide-map.pdf",
    "penitencia-creek-parkway_geo.json": "penitencia-creek-guide-map.pdf",
    "sanborn_geo.json": "sanborn-guide-map.pdf",
    "santa-teresa_geo.json": "santa-teresa-guide-map.pdf",
    "stevens-creek_geo.json": "stevens-creek-guide-map.pdf",
    "upper-stevens-creek_geo.json": "upper-stevens-creek-guide-map.pdf",
    "uvas-canyon_geo.json": "uvas-canyon-guide-map.pdf",
    "vasona-lake_geo.json": "Vasona Guide Map.pdf",
    "villa-montalvo_geo.json": "villa-montalvo-guide-map.pdf",
}


@lru_cache(maxsize=1)
def _sheets():
    out = []
    for geo_name, pdf_name in SHEETS.items():
        geo_path = os.path.join(GEO_DIR, geo_name)
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        geo = atlas_scc.load_geo(geo_path)
        fwd = geo["affine_pixel_to_lnglat"]
        a, b, _ = fwd["lng"]
        d, e, _ = fwd["lat"]
        w, h = geo["imageWidth"], geo["imageHeight"]
        area = abs(a * e - b * d) * w * h
        out.append({"geo_path": geo_path, "pdf_path": pdf_path, "fwd": fwd, "w": w, "h": h, "area": area})
    return out


def _on_sheet_fraction(sheet, coords):
    inside = 0
    for lon, lat in coords:
        px, py = atlas_scc.ll_to_px(sheet["fwd"], lon, lat)
        if 0 <= px <= sheet["w"] and 0 <= py <= sheet["h"]:
            inside += 1
    return inside / len(coords)


def pick_sheet(coords):
    sample = coords[:: max(1, len(coords) // MAX_SAMPLE)]
    best, best_key = None, None
    for sheet in _sheets():
        frac = _on_sheet_fraction(sheet, sample)
        if frac < MIN_ON_SHEET:
            continue
        key = (frac, -sheet["area"])
        if best_key is None or key > best_key:
            best, best_key = sheet, key
    return best


class MapUnavailable(Exception):
    pass


def _park_slug(pdf_path):
    stem = os.path.splitext(os.path.basename(pdf_path))[0].lower()
    stem = stem.replace("guide map", "").replace("guide-map", "")
    return "-".join(t for t in "".join(ch if ch.isalnum() else " " for ch in stem).split())


def _validate(data):
    if not data or data[:4] != b"%PDF":
        raise ValueError("Generated file is not a PDF")
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.page_count < 1:
            raise ValueError("Generated PDF has no pages")
    return data


def _render_mapbox(coords, name, distance_mi, gain_ft, out_path):
    token = os.environ.get("MAPBOX_TOKEN", "").strip()
    if not token:
        raise MapUnavailable("Fallback map unavailable: MAPBOX_TOKEN is not configured")
    try:
        atlas_mapbox.render_mapbox_pdf(coords, name, distance_mi, gain_ft, token, out_path)
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        raise MapUnavailable(f"Fallback map unavailable: Mapbox request failed ({status or type(e).__name__})") from None


def render_route_map(gpx_path, name, distance_mi, gain_ft):
    coords, _ = atlas_scc.parse_gpx(gpx_path)
    if not coords:
        raise ValueError(f"No track points in {gpx_path}")
    sheet = pick_sheet(coords)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "route.pdf")
        if sheet:
            try:
                atlas_scc.render_scc_pdf(
                    coords, name, distance_mi, gain_ft,
                    sheet["geo_path"], sheet["pdf_path"], out_path,
                )
                with open(out_path, "rb") as f:
                    return _validate(f.read()), "overlay", _park_slug(sheet["pdf_path"])
            except Exception:
                if os.path.exists(out_path):
                    os.remove(out_path)
        _render_mapbox(coords, name, distance_mi, gain_ft, out_path)
        with open(out_path, "rb") as f:
            return _validate(f.read()), "fallback", None


def render_route_pdf(gpx_path, name, distance_mi, gain_ft):
    return render_route_map(gpx_path, name, distance_mi, gain_ft)[0]
