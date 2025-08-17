#!/usr/bin/env python3
"""
Visualize YOLO/COCO predictions with channel-aware overlays.

- Reads a COCO JSON (with optional custom keys: "hf_channels", "confidence").
- Filters annotations where "hf_channels" has exactly one element.
- For each kept annotation:
  * Tries to load the original TIFF corresponding to the image and extract the
    specified channel (1-based index) as a grayscale background.
    - If unavailable, falls back to the JPEG referenced by COCO.
  * Draws the bounding box in magenta and the polygon(s) in magenta with ~30% opacity.
  * Renders a short label "class conf" in magenta.
  * Saves one JPEG per annotation into the output folder.

Usage:
    python visualize_predictions.py \
        --coco path/to/coco.json \
        --images_dir path/to/images \
        --out_dir path/to/viz_out \
        [--tiff_dir path/to/tiffs]

Notes:
- TIFF inference: the script tries to guess the TIFF name by stripping suffixes
  from the JPEG basename at the first underscore "_". You can adapt the logic
  in `guess_tiff_path()` if needed.
- Channel indexing: assumes hf_channels contains 1-based indices.
- If COCO coordinates differ from the image pixel size, they are scaled.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import tifffile
import cv2


# ------------- Geometry helpers ------------------------------------------------

def scale_coords(coords: List[float], sx: float, sy: float) -> List[float]:
    """Scale flattened [x1,y1,x2,y2,...] by (sx, sy)."""
    out = coords[:]
    out[0::2] = [v * sx for v in out[0::2]]
    out[1::2] = [v * sy for v in out[1::2]]
    return out


def clamp_box_xyxy(box: List[float], W: int, H: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(x1, W - 1))
    y1 = max(0.0, min(y1, H - 1))
    x2 = max(0.0, min(x2, W - 1))
    y2 = max(0.0, min(y2, H - 1))
    return [x1, y1, x2, y2]


# ------------- TIFF handling ---------------------------------------------------

def guess_tiff_path(images_dir: str, tiff_dir: Optional[str], image_filename: str) -> Optional[str]:
    """
    Guess the TIFF path from an image filename.
    Strategy: strip suffix at first '_' and try <stem>.tif / <stem>.tiff in tiff_dir.
    """
    if not tiff_dir:
        return None
    stem = os.path.splitext(os.path.basename(image_filename))[0]
    stem = stem.split("_")[0] if "_" in stem else stem
    cand1 = os.path.join(tiff_dir, stem + ".tif")
    cand2 = os.path.join(tiff_dir, stem + ".tiff")
    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    return None


def extract_channel(arr: np.ndarray, ch_index_1based: int) -> np.ndarray:
    """
    Try to extract a single channel (1-based) from a TIFF array with unknown layout.
    Heuristics:
      - If arr.ndim == 2: return as is.
      - Prefer last axis if it's small and >= ch.
      - Else prefer first axis if >= ch.
      - Else try middle axis.
    Returns uint8 2D image (contrast-normalized).
    """
    ch = max(1, int(ch_index_1based)) - 1

    a = arr
    # Squeeze singular dims to simplify (e.g., (1,C,H,W) -> (C,H,W))
    a = np.squeeze(a)

    if a.ndim == 2:
        plane = a
    elif a.ndim == 3:
        axes = list(a.shape)  # e.g., (H,W,C) or (C,H,W) or (H,C,W)
        # choose axis whose size >= ch+1 and is smallest among candidates (likely channels)
        candidates = [(ax_i, axes[ax_i]) for ax_i in range(3) if axes[ax_i] >= ch + 1]
        if not candidates:
            # fallback: take last axis index 0
            plane = a[..., 0]
        else:
            chan_axis = min(candidates, key=lambda t: t[1])[0]
            slicer = [slice(None)] * 3
            slicer[chan_axis] = ch
            plane = a[tuple(slicer)]
    elif a.ndim == 4:
        # Try to find a channel axis (size >= ch+1 and smaller than typical spatial dims)
        axes = list(a.shape)  # e.g., (Z,C,H,W) or (C,Z,H,W) or (Z,H,W,C)
        H_like = max(axes)  # rough heuristic
        candidates = [(ax_i, axes[ax_i]) for ax_i in range(4)
                      if axes[ax_i] >= ch + 1 and axes[ax_i] < H_like]
        if candidates:
            chan_axis = min(candidates, key=lambda t: t[1])[0]
            slicer = [0] * 4  # take first index for non-channel axes
            for k in range(4):
                slicer[k] = ch if k == chan_axis else 0
            plane = a[tuple(slicer)]
        else:
            # fallback to last axis
            plane = a[..., 0]
    else:
        # Too many dims; just flatten leading dims and pick last axis as channel
        while a.ndim > 3:
            a = a[0]
        return extract_channel(a, ch + 1)

    # Normalize to 8-bit for visualization
    plane = np.asarray(plane, dtype=np.float32)
    p1, p99 = np.percentile(plane, [1, 99])
    if p99 <= p1:
        p1, p99 = float(np.min(plane)), float(np.max(plane) + 1e-6)
    plane = np.clip((plane - p1) / (p99 - p1 + 1e-9), 0, 1)
    plane = (plane * 255.0).astype(np.uint8)
    return plane


# ------------- Drawing ---------------------------------------------------------

MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
ALPHA = int(0.3 * 255)  # ~30%

def draw_annotation(
    base_img: Image.Image,
    bbox_xyxy: List[float],
    segs: List[List[float]],
    label: str,
    stroke_width: int = 2
) -> Image.Image:
    """
    Draw bbox and polygons onto a copy of base_img (RGB).
    Polygons are filled with magenta at ~30% opacity; bbox and text are magenta.
    """
    W, H = base_img.size
    canvas = base_img.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # BBox
    x1, y1, x2, y2 = clamp_box_xyxy(bbox_xyxy, W, H)
    draw.rectangle([x1, y1, x2, y2], outline=MAGENTA, width=stroke_width)

    # Label (simple)
    try:
        font = ImageFont.truetype("arial.ttf", 22) or ImageFont.load_default()
    except Exception:
        font = None
    text = label
    tw, th = draw.textlength(text, font=font), 10 if font is None else font.size + 2
    tx, ty = max(0, int(x1)), max(0, int(y1) - th - 2)
    # Optional background for readability
    draw.rectangle([tx, ty, tx + tw + 4, ty + th], fill=(0, 0, 0))
    draw.text((tx + 2, ty + 1), text, fill=MAGENTA, font=font)

    # Polygons with alpha: draw on an RGBA overlay then composite
    if segs:
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay, "RGBA")
        for seg in segs:
            if len(seg) >= 6:
                pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                odraw.polygon(pts, outline=MAGENTA + (255,), fill=MAGENTA + (ALPHA,))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")



    return canvas


# ------------- COCO I/O --------------------------------------------------------

def load_coco(path: str) -> Tuple[Dict, Dict[int, str], Dict[int, Dict]]:
    with open(path, "r") as f:
        coco = json.load(f)
    id_to_name = {}
    for c in coco.get("categories", []):
        id_to_name[int(c["id"])] = c["name"]
    images_by_id = {int(img["id"]): img for img in coco.get("images", [])}
    return coco, id_to_name, images_by_id


# ------------- Main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize COCO predictions with channel-aware overlays.")
    ap.add_argument("--coco", required=True, help="Path to coco.json")
    ap.add_argument("--images_dir", required=True, help="Directory containing the referenced images")
    ap.add_argument("--out_dir", required=True, help="Output directory for visualizations (JPEG)")
    ap.add_argument("--tiff_dir", default=None, help="Directory of original TIFFs (optional, recommended)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    coco, id_to_name, images_by_id = load_coco(args.coco)
    anns = coco.get("annotations", [])
    print(f"[info] Loaded {len(anns)} annotations.")

    kept = [a for a in anns if isinstance(a.get("hf_channels"), list) and len(a["hf_channels"]) == 1]
    print(f"[info] Kept {len(kept)} annotations with exactly one hf_channel.")

    # Prepare per-image scaling if needed (COCO width/height vs actual image)
    for ann in kept:
        img_info = images_by_id.get(int(ann["image_id"]))
        if not img_info:
            print(f"[warn] Missing image for annotation id={ann.get('id')}; skipping.")
            continue

        img_path = os.path.join(args.images_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            print(f"[warn] Image not found: {img_path}; skipping.")
            continue

        # Try to prepare grayscale background from TIFF channel
        bg = None
        tiff_path = img_path #guess_tiff_path(args.images_dir, args.tiff_dir, img_info["file_name"])
        if tiff_path and os.path.isfile(tiff_path):
            try:
                tif = tifffile.imread(tiff_path)
                ch = int(ann["hf_channels"][0])  # 1-based
                plane = extract_channel(tif, ch)
                bg = Image.fromarray(plane)
                bg = ImageOps.grayscale(bg) # in case there are indexed colors
            except Exception as e:
                print(f"[warn] TIFF load/extract failed for {tiff_path}: {e}")

        if bg is None:
            # Fallback to the JPEG used in COCO
            bg = Image.open(img_path).convert("RGB")

        W_img, H_img = bg.size
        W_coco = int(img_info.get("width", W_img))
        H_coco = int(img_info.get("height", H_img))
        sx = W_img / float(W_coco) if W_coco else 1.0
        sy = H_img / float(H_coco) if H_coco else 1.0

        # Box and polygons (scale if needed)
        x, y, w, h = ann["bbox"]
        bbox_xyxy = [x, y, x + w, y + h]
        bbox_xyxy = scale_coords(bbox_xyxy, sx, sy)

        segs = []
        for seg in ann.get("segmentation", []):
            if isinstance(seg, list) and len(seg) >= 6:
                segs.append(scale_coords(seg, sx, sy))

        cls_id = int(ann["category_id"])
        cls_name = id_to_name.get(cls_id, f"class_{cls_id}")
        conf = ann.get("confidence")
        conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        label = f"{cls_name} ({conf_txt})"

        vis = draw_annotation(bg, bbox_xyxy, segs, label, stroke_width=2)

        base = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
        out_name = f"{base}_ann{ann['id']}_ch{ann['hf_channels'][0]}.jpg"
        out_path = os.path.join(args.out_dir, out_name)
        vis.save(out_path, "JPEG", quality=100)
        print(f"[ok] Wrote {out_path}")

    print("[done]")


if __name__ == "__main__":
    main()

