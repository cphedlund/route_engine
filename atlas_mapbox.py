#!/usr/bin/env python3
"""
atlas_mapbox.py — the universal backup map. Renders any route on a Mapbox
outdoors (topo) basemap and wraps it in a plain printable PDF with a title card.
Works for EVERY route, including those with no official SCC sheet.

Requires network + a Mapbox token (public pk. token is fine). Run locally.

Core entry point:
  render_mapbox_pdf(coords, name, distance_mi, gain_ft, token, out_path)
    coords: list of [lng, lat]
"""
import argparse, json, math, io, urllib.parse
import requests
import fitz  # PyMuPDF

ACCENT = (0.910, 0.584, 0.278)
CARD_BG = (0.988, 0.992, 0.980)
CARD_LN = (0.824, 0.843, 0.796)
INK = (0.106, 0.122, 0.110)
STYLE = "outdoors-v12"


def encode_polyline(latlng):
    """Google polyline algorithm (precision 5). Input: list of (lat, lng)."""
    out = []
    plat = plng = 0
    for lat, lng in latlng:
        ilat, ilng = round(lat * 1e5), round(lng * 1e5)
        for v in (ilat - plat, ilng - plng):
            v <<= 1
            if v < 0:
                v = ~v
            while v >= 0x20:
                out.append(chr((0x20 | (v & 0x1F)) + 63))
                v >>= 5
            out.append(chr(v + 63))
        plat, plng = ilat, ilng
    return "".join(out)


def simplify(coords, max_pts=480):
    """Thin points evenly if the encoded path would be very long (URL safety)."""
    if len(coords) <= max_pts:
        return coords
    step = len(coords) / max_pts
    out = [coords[int(i * step)] for i in range(max_pts)]
    out[-1] = coords[-1]
    return out


def static_url(coords, token, w=1200, h=900):
    poly = encode_polyline([(lat, lng) for lng, lat in simplify(coords)])
    overlay = "path-5+e89547-1(" + urllib.parse.quote(poly, safe="") + ")"
    return (f"https://api.mapbox.com/styles/v1/mapbox/{STYLE}/static/"
            f"{overlay}/auto/{w}x{h}@2x?access_token={token}&padding=60")


def render_mapbox_pdf(coords, name, distance_mi, gain_ft, token, out_path):
    url = static_url(coords, token)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    img_bytes = r.content

    src = fitz.open(stream=img_bytes, filetype="png")
    rect = src[0].rect
    pdf = fitz.open()
    page = pdf.new_page(width=rect.width, height=rect.height)
    page.insert_image(page.rect, stream=img_bytes)

    # title card, top-right corner (topo base fills the frame, no legend panel)
    title = name
    stats = f"{distance_mi:.1f} mi   \u2022   {gain_ft:,.0f} ft gain"
    fs_t, fs_s = 15, 11
    tw = max(fitz.get_text_length(title, fontname="hebo", fontsize=fs_t),
             fitz.get_text_length(stats, fontname="helv", fontsize=fs_s))
    padx, pady, margin = 11, 9, 14
    bw = tw + padx * 2 + 5
    bh = fs_t + fs_s + pady * 2 + 6
    x1 = page.rect.width - margin
    x0 = x1 - bw
    y0 = margin
    y1 = y0 + bh
    shp = page.new_shape()
    shp.draw_rect(fitz.Rect(x0, y0, x1, y1)); shp.finish(color=CARD_LN, fill=CARD_BG, width=0.8)
    shp.draw_rect(fitz.Rect(x0, y0, x0 + 4, y1)); shp.finish(color=ACCENT, fill=ACCENT)
    shp.commit()
    page.insert_text(fitz.Point(x0 + padx + 5, y0 + pady + fs_t - 2), title,
                     fontname="hebo", fontsize=fs_t, color=INK)
    page.insert_text(fitz.Point(x0 + padx + 5, y0 + pady + fs_t + fs_s + 2), stats,
                     fontname="helv", fontsize=fs_s, color=(0.42, 0.44, 0.40))
    pdf.save(out_path, garbage=4, deflate=True)
    pdf.close()
    return url


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpx")
    ap.add_argument("--coords", help="JSON file with a [[lng,lat],...] array")
    ap.add_argument("--token", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Route")
    ap.add_argument("--distance", type=float, default=0.0)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--print-url", action="store_true", help="just print the static URL")
    a = ap.parse_args()

    if a.gpx:
        import xml.etree.ElementTree as ET
        root = ET.parse(a.gpx).getroot()
        ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        tag = lambda t: f"{{{ns}}}{t}" if ns else t
        coords = [[float(tp.get("lon")), float(tp.get("lat"))] for tp in root.iter(tag("trkpt"))]
    else:
        coords = json.load(open(a.coords))

    if a.print_url:
        print(static_url(coords, a.token))
    else:
        url = render_mapbox_pdf(coords, a.title, a.distance, a.gain, a.token, a.out)
        print("wrote", a.out)
