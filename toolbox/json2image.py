#!/usr/bin/env python3
"""
Visualize predictions: one PNG per category.

For each TIFF in --tiff_dir:
  - collect all *.coco.json in --coco_dir whose basename matches the TIFF,
  - merge annotations & categories,
  - render ONE image per category with all its annotations drawn.

Background: strict grayscale from the TIFF (2D plane or first plane).
Overlays: bbox + polygons. Color is magenta by default, or confidence-coded with --viridis.
Stroke and font size scale with image width.

Usage:
  python visualize_predictions.py \
      --tiff_dir /path/to/tiffs \
      --coco_dir /path/to/jsons \
      --out_dir  /path/to/out \
      [--viridis] [--jpeg]
"""

import os, re, glob, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import tifffile
import matplotlib
import matplotlib.colors as mcolors

MAGENTA = (255, 0, 255)
ALPHA = int(0.30 * 255)  # ~30% fill opacity

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9+._-]+", "_", name)

def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    return [max(0, min(x1, W-1)),
            max(0, min(y1, H-1)),
            max(0, min(x2, W-1)),
            max(0, min(y2, H-1))]

def scale_coords(coords, sx: float, sy: float):
    out = coords[:]
    out[0::2] = [v * sx for v in out[0::2]]
    out[1::2] = [v * sy for v in out[1::2]]
    return out

def load_all_coco_for_base(base: str, coco_dir: str):
    """
    Return (annotations, id_to_name, ref_w, ref_h).
    ref_w/ref_h are pulled from the first JSON that provides them, else None.
    """
    pattern = os.path.join(coco_dir, base + "*.json")
    files = sorted(glob.glob(pattern))
    anns = []
    id_to_name = {}
    ref_w = ref_h = None
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)
        # categories → id_to_name
        for c in d.get("categories", []):
            cid = int(c["id"])
            nm = c.get("name", f"class_{cid}")
            id_to_name[cid] = nm
        # reference dims (first available)
        imgs = d.get("images", [])
        if imgs and (ref_w is None or ref_h is None):
            # take first entry
            iw = imgs[0].get("width")
            ih = imgs[0].get("height")
            if isinstance(iw, (int, float)) and isinstance(ih, (int, float)):
                ref_w, ref_h = float(iw), float(ih)
        # collect annotations
        anns.extend(d.get("annotations", []))
    return anns, id_to_name, ref_w, ref_h

def draw_annotation(canvas_rgb: Image.Image,
                    bbox_xyxy,
                    segs,
                    label: str,
                    color: tuple,
                    stroke: int) -> Image.Image:
    """Draw bbox + filled polygons (30% alpha) + label."""
    W, H = canvas_rgb.size
    draw = ImageDraw.Draw(canvas_rgb)
    # bbox
    x1, y1, x2, y2 = clamp_box_xyxy(bbox_xyxy, W, H)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)

    # label (bold-ish)
    try:
        # scale font with width
        font_size = max(stroke, int(canvas_rgb.width // 50))
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    tw = draw.textlength(label, font=font)
    th = (font.size + 4) if hasattr(font, "size") else 14
    tx, ty = max(0, int(x1)), max(0, int(y1) - th - 2)
    draw.rectangle([tx, ty, tx + tw + 6, ty + th], fill=(0, 0, 0))
    for dx in (0, 1):
        for dy in (0, 1):
            draw.text((tx + 3 + dx, ty + 2 + dy), label, fill=color, font=font)

    # polygons via overlay
    if segs:
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay, "RGBA")
        for seg in segs:
            if isinstance(seg, list) and len(seg) >= 6:
                pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                odraw.polygon(pts, outline=color + (255,), fill=color + (ALPHA,))
        canvas_rgb = Image.alpha_composite(canvas_rgb.convert("RGBA"), overlay).convert("RGB")

    return canvas_rgb

def main():
    ap = argparse.ArgumentParser(description="Render one visualization per category from COCO predictions.")
    ap.add_argument("--tiff_dir", required=True)
    ap.add_argument("--coco_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--viridis", action="store_true", help="Color-code by confidence")
    ap.add_argument("--jpeg", action="store_true", help="Save JPEG instead of PNG")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # colormap for confidence
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.colormaps.get_cmap("viridis")

    tiff_paths = sorted(glob.glob(os.path.join(args.tiff_dir, "*.tif")) +
                        glob.glob(os.path.join(args.tiff_dir, "*.tiff")))
    for tif in tiff_paths:
        base = os.path.splitext(os.path.basename(tif))[0]
        anns, id_to_name, ref_w, ref_h = load_all_coco_for_base(base, args.coco_dir)
        if not anns:
            continue

        # TIFF → background grayscale (2D plane or first plane)
        arr = tifffile.imread(tif)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            plane = arr
        elif arr.ndim >= 3:
            # pick first plane safely
            plane = arr[(0,) * (arr.ndim - 2)]
        else:
            plane = arr
        # normalize to uint8 if needed
        if plane.dtype != np.uint8:
            p1, p99 = np.percentile(plane.astype(np.float32), [1, 99])
            if p99 <= p1:
                p1, p99 = float(np.min(plane)), float(np.max(plane) + 1e-6)
            plane = np.clip((plane - p1) / (p99 - p1 + 1e-9), 0, 1)
            plane = (plane * 255.0).astype(np.uint8)
        bg = ImageOps.grayscale(Image.fromarray(plane)).convert("RGB")
        W, H = bg.size
        stroke = max(1, W // 500)

        # If COCO dims provided, compute scale
        if ref_w and ref_h:
            sx = W / float(ref_w)
            sy = H / float(ref_h)
        else:
            sx = sy = 1.0

        # group by (category, channel)
        by_cat_ch = {}
        for a in anns:
            cid = int(a["category_id"])
            ch = int(a.get("hf_channels", [0])[0])  # default to channel 0 if missing
            by_cat_ch.setdefault((cid, ch), []).append(a)

        for (cid, ch), cat_anns in sorted(by_cat_ch.items()):
            cls_name = id_to_name.get(cid, f"class_{cid}")

            # pick the right channel plane
            if arr.ndim == 3:
                plane = arr[ch]
            else:
                plane = arr  # fallback if no channels

            # normalize plane → uint8 grayscale background
            if plane.dtype != np.uint8:
                p1, p99 = np.percentile(plane.astype(np.float32), [1, 99])
                if p99 <= p1:
                    p1, p99 = float(np.min(plane)), float(np.max(plane) + 1e-6)
                plane = np.clip((plane - p1) / (p99 - p1 + 1e-9), 0, 1)
                plane = (plane * 255.0).astype(np.uint8)
            bg = ImageOps.grayscale(Image.fromarray(plane)).convert("RGB")

            # draw annotations for this class/channel...
            canvas = bg.copy()
            for a in cat_anns:
                x, y, w, h = a["bbox"]
                bbox_xyxy = scale_coords([x, y, x + w, y + h], sx, sy)
                segs = [scale_coords(seg, sx, sy)
                        for seg in a.get("segmentation", [])
                        if isinstance(seg, list) and len(seg) >= 6]
                conf = float(a.get("confidence", 0.0))
                label = f"{cls_name[:2]}. ({conf:.2f})"
                color = MAGENTA if not args.viridis else tuple(int(255*v) for v in cmap(norm(conf))[:3])
                canvas = draw_annotation(canvas, bbox_xyxy, segs, label, color, stroke)

            out_name = f"{base}_{sanitize(cls_name)}_ch{ch}" + (".jpg" if args.jpeg else ".png")
            canvas.save(os.path.join(args.out_dir, out_name))
            print(f"[ok] {os.path.basename(tif)} → {out_name}")

if __name__ == "__main__":
    main()

