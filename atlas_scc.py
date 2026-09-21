#!/usr/bin/env python3
"""
atlas_scc.py — draw a route as a VECTOR overlay on the untouched official SCC
park map PDF. Output is an exact copy of the official sheet (crisp at any zoom,
prints at any size) with the route highlighted. Plain PDF, no GPS layer.

Core entry point:
  render_scc_pdf(coords, name, distance_mi, gain_ft, geojson_path, map_pdf_path, out_path)
    coords: list of [lng, lat]

CLI (for testing with a GPX):
  python atlas_scc.py --gpx R.gpx --geojson P.geo.json --map P.pdf --out O.pdf --title "Name"
"""
import argparse, json, math
import numpy as np
import fitz  # PyMuPDF

ACCENT = (0.910, 0.584, 0.278)   # #E89547
CASING = (1, 1, 1)
START  = (0.306, 0.490, 0.329)
END    = (0.710, 0.314, 0.227)
INK    = (0.106, 0.122, 0.110)
CARD_BG = (0.988, 0.992, 0.980)
CARD_LN = (0.824, 0.843, 0.796)


def load_geo(path):
    return json.load(open(path))


def ll_to_px(fwd, lon, lat):
    """lng/lat -> pixel via analytic inverse of the forward affine (exact, stable)."""
    a, b, c = fwd["lng"]
    d, e, f = fwd["lat"]
    det = a * e - b * d
    px = (e * (lon - c) - b * (lat - f)) / det
    py = (-d * (lon - c) + a * (lat - f)) / det
    return px, py


def detect_map_box_px(page, img_w, img_h):
    """Render the page and find the right/top edge of the cartographic map area
    (vs the legend panel) via the white gutter. Returns (top_px, right_px)."""
    zoom = img_w / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    a = a[:, :, :3]
    W, H = pix.width, pix.height
    white = (a[:, :, 0] > 240) & (a[:, :, 1] > 240) & (a[:, :, 2] > 240)
    colw = np.convolve(white.mean(axis=0), np.ones(41) / 41, mode="same")
    thr, need, run, right = 0.45, 120, 0, W
    for x in range(int(0.30 * W), W):
        run = run + 1 if colw[x] > thr else 0
        if run >= need:
            right = x - need
            break
    roww = np.convolve(white[:, :right].mean(axis=1), np.ones(41) / 41, mode="same")
    top = 0
    for y in range(H):
        if roww[y] < 0.55:
            top = y
            break
    return top * (W / img_w and 1.0), right, W  # right in render px; W == img_w


def render_scc_pdf(coords, name, distance_mi, gain_ft, geojson_path, map_pdf_path, out_path):
    geo = load_geo(geojson_path)
    fwd = geo["affine_pixel_to_lnglat"]
    img_w, img_h = geo["imageWidth"], geo["imageHeight"]

    doc = fitz.open(map_pdf_path)
    page = doc[0]
    sx = page.rect.width / img_w     # points per image-pixel
    sy = page.rect.height / img_h

    # route in page points (fitz page space is top-left origin, same as the image)
    pts = []
    inb = 0
    for lon, lat in coords:
        px, py = ll_to_px(fwd, lon, lat)
        if 0 <= px <= img_w and 0 <= py <= img_h:
            inb += 1
        pts.append(fitz.Point(px * sx, py * sy))

    # route: white casing then accent line
    shp = page.new_shape()
    shp.draw_polyline(pts)
    shp.finish(color=CASING, width=5.4, closePath=False, lineCap=1, lineJoin=1)
    shp.commit()
    shp = page.new_shape()
    shp.draw_polyline(pts)
    shp.finish(color=ACCENT, width=2.8, closePath=False, lineCap=1, lineJoin=1)
    shp.commit()

    # start / end markers
    for p, col in ((pts[0], START), (pts[-1], END)):
        shp = page.new_shape()
        shp.draw_circle(p, 6.2)
        shp.finish(color=CASING, fill=CASING)
        shp.draw_circle(p, 4.6)
        shp.finish(color=col, fill=col)
        shp.commit()

    # title card in the upper-right of the map area (not the legend panel)
    top_px, right_px, render_w = detect_map_box_px(page, img_w, img_h)
    right_pt = right_px * (page.rect.width / render_w)
    top_pt = 0 * sy + page.rect.height * 0  # top of map ~ page top
    title = name
    stats = f"{distance_mi:.1f} mi   \u2022   {gain_ft:,.0f} ft gain"
    fs_t, fs_s = 15, 11
    tw = max(fitz.get_text_length(title, fontname="hebo", fontsize=fs_t),
             fitz.get_text_length(stats, fontname="helv", fontsize=fs_s))
    padx, pady, margin = 11, 9, 12
    bw = tw + padx * 2 + 5
    bh = fs_t + fs_s + pady * 2 + 6
    x1 = right_pt - margin
    x0 = x1 - bw
    y0 = margin
    y1 = y0 + bh
    card = fitz.Rect(x0, y0, x1, y1)
    shp = page.new_shape()
    shp.draw_rect(card)
    shp.finish(color=CARD_LN, fill=CARD_BG, width=0.8)
    shp.draw_rect(fitz.Rect(x0, y0, x0 + 4, y1))
    shp.finish(color=ACCENT, fill=ACCENT)
    shp.commit()
    page.insert_text(fitz.Point(x0 + padx + 5, y0 + pady + fs_t - 2),
                     title, fontname="hebo", fontsize=fs_t, color=INK)
    page.insert_text(fitz.Point(x0 + padx + 5, y0 + pady + fs_t + fs_s + 2),
                     stats, fontname="helv", fontsize=fs_s, color=(0.42, 0.44, 0.40))

    doc.save(out_path, garbage=4, deflate=True)
    doc.close()
    return inb, len(coords)


# ---- helpers for CLI testing with a GPX ----
def parse_gpx(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    tag = lambda t: f"{{{ns}}}{t}" if ns else t
    coords, eles = [], []
    for tp in root.iter(tag("trkpt")):
        coords.append([float(tp.get("lon")), float(tp.get("lat"))])
        ele = tp.find(tag("ele"))
        if ele is not None:
            eles.append(float(ele.text))
    return coords, eles


def stats_from(coords, eles):
    R = 6371000.0; p = math.pi / 180
    dist = 0.0
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        dlat = (b[1] - a[1]) * p; dlon = (b[0] - a[0]) * p
        h = math.sin(dlat / 2) ** 2 + math.cos(a[1] * p) * math.cos(b[1] * p) * math.sin(dlon / 2) ** 2
        dist += 2 * R * math.asin(math.sqrt(h))
    gain = sum(max(0, eles[i] - eles[i - 1]) for i in range(1, len(eles)))
    return dist / 1609.34, gain * 3.28084


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpx", required=True)
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Route")
    a = ap.parse_args()
    coords, eles = parse_gpx(a.gpx)
    mi, ft = stats_from(coords, eles)
    inb, tot = render_scc_pdf(coords, a.title, mi, ft, a.geojson, a.map, a.out)
    print(f"on-sheet {inb}/{tot} ({100*inb/tot:.0f}%) · {mi:.1f} mi · {ft:,.0f} ft · wrote {a.out}")
