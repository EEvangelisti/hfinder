#!/usr/bin/env python3

"""
Render one PNG per category from COCO predictions aligned to TIFF stacks.

**Pipeline**
- For each TIFF in ``--tiff_dir``:
  - collect all matching ``*.json`` COCO files from ``--annotations`` (basename match),
  - merge their categories and annotations,
  - render **one** image per (category, channel) with bounding boxes, polygons and labels.

**Background**
Grayscale from the TIFF (2D plane or first plane). Overlays include bounding boxes and
filled polygons. Color is magenta by default or confidence-coded via a matplotlib colormap
(e.g. ``viridis``) when ``--palette`` is provided. Stroke and font size scale with image width.

**Usage**
.. code-block:: bash

    python annot2images.py \
        --tiff_dir /path/to/tiffs \
        --annotations /path/to/jsons \
        --out_dir /path/to/out \
        [--palette viridis] [--no_labels] [--no_bounding_boxes] \
        [--font_file DejaVuSans.ttf] [--font_size 18]

"""

import re
import json
import argparse
import numpy as np
import tifffile
import matplotlib
from pathlib import Path
from itertools import chain
from collections import defaultdict
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont, ImageOps


CYAN = (0, 255, 255)
ALPHA30 = int(0.30 * 255)
SETTINGS = None

ARGLIST = {
    "-t": {
        "long": "--tiff_dir",
        "config": {
            "default": ".",
            "help": "Folder containing TIFF files"
        }
    },
    "-a": {
        "long": "--annotations",
        "config": {
            "default": ".",
            "help": "Folder containing COCO JSON files"
        }
    },
    "-o": {
        "long": "--out_dir",
        "config": {
            "default": ".",
            "help": "Output directory for PNG files"
        }
    },
    "-lab": {
        "long": "--no_labels",
        "config": {
            "action": "store_true",
            "help": "Do not display labels and confidence values - useful on crowded images"
        }
    },
    "-box": {
        "long": "--no_bounding_boxes",
        "config": {
            "action": "store_true",
            "help": "Do not display bounding boxes around polygons"
        }
    },
    "-pal": {
        "long": "--palette",
        "config": {
            "default": None,
            "help": "Matplotlib colormap used to encode confidence values, e.g. viridis, plasma, cool, etc."
        }
    },
    "-ttf": {
        "long": "--font_file",
        "config": {
            "default": "DejaVuSans.ttf",
            "help": "Font used to write labels and confidence values"
        }
    },
    "-sz": {
        "long": "--font_size",
        "config": {
            "default": "proportional",
            "help": "Font size for labels and confidence values"
        }
    },
    "-long": {
        "long": "--long_labels",
        "config": {
            "action": "store_true",
            "help": "Do not abbreviate label names"
        }
    },
    "-cat": {
        "long": "--category",
        "config": {
            "default": "*",
            "help": "Process the given category only"
        }
    },
}


def sanitize(name):
    """
    Make a filesystem-friendly token from an arbitrary string.

    Replaces any character outside ``[A-Za-z0-9+._-]`` with an underscore.

    :param name: Raw string to sanitize.
    :type name: str
    :returns: Sanitized string safe for filenames.
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9+._-]+", "_", name)



def clamp_box_xyxy(box, W, H):
    """
    Clamp an axis-aligned box to image bounds in (x1, y1, x2, y2) form.

    :param box: Box coordinates ``[x1, y1, x2, y2]`` (can be floats).
    :type box: list | tuple
    :param W: Image width in pixels.
    :type W: int
    :param H: Image height in pixels.
    :type H: int
    :returns: Clamped box ``[x1, y1, x2, y2]`` within ``[0..W-1] × [0..H-1]``.
    :rtype: list[int]
    """
    x1, y1, x2, y2 = box
    return [int(max(0, min(x1, W-1))),
            int(max(0, min(y1, H-1))),
            int(max(0, min(x2, W-1))),
            int(max(0, min(y2, H-1)))]



def load_all_coco_for_base(base, coco_dir):
    """
    Load and merge all COCO JSONs whose basename starts with ``base`` from ``coco_dir``.

    Aggregates:
      - ``categories`` → a mapping ``category_id → category_name``
      - ``annotations`` → a flat list across all files

    Note: this function currently returns only annotations and the ``id_to_name`` mapping.

    :param base: Basename (without extension) of the reference TIFF.
    :type base: str
    :param coco_dir: Directory containing COCO ``*.json`` files.
    :type coco_dir: str
    :returns: Tuple ``(annotations, id_to_name)``.
    :rtype: tuple[list[dict], dict[int, str]]
    """
    files = sorted(Path(coco_dir).glob(f"{base}_{SETTINGS.category}.json"))
    anns = []
    id_to_name = {}
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)
        # categories → id_to_name
        for c in d.get("categories", []):
            cid = int(c["id"])
            nm = c.get("name", f"class_{cid}")
            id_to_name[cid] = nm
        # collect annotations
        anns.extend(d.get("annotations", []))
    return anns, id_to_name



def normalize_to_uint8(arr):
    """
    Normalize an image array to uint8 (0–255) for display/export.

    - If dtype is already uint8 → returned as-is.
    - If dtype is uint16/float → scaled to [0,255] based on min–max.
    - If max == min → flat image filled with zeros.

    :param arr: Input 2D image (H, W).
    :type arr: numpy.ndarray
    :returns: Normalized 8-bit image.
    :rtype: numpy.ndarray
    """
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32, copy=False)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr.astype(np.uint8)



def draw_annotation(img, bbox_xyxy, segs, label, color, stroke):
    """
    Draw a bounding box, optional filled polygons (30% alpha), and a label on an image.

    - Bounding box is skipped when ``SETTINGS.no_bounding_boxes`` is True.
    - Label is drawn on a black rectangle that clamps to image bounds.
    - Polygons are composited on an RGBA overlay to preserve alpha.

    :param img: PIL image (RGB).
    :type img: PIL.Image.Image
    :param bbox_xyxy: Bounding box as ``[x1, y1, x2, y2]``.
    :type bbox_xyxy: list | tuple
    :param segs: COCO polygon segmentations (list of flat ``[x0, y0, x1, y1, ...]`` lists).
    :type segs: list
    :param label: Text label (e.g., ``"Cl. (0.92)"``) or ``None`` to skip.
    :type label: str | None
    :param color: RGB tuple for strokes/fills, e.g. ``(0, 255, 255)``.
    :type color: tuple[int, int, int]
    :param stroke: Stroke width in pixels.
    :type stroke: int
    :returns: The image after drawing all elements.
    :rtype: PIL.Image.Image
    """
    W, H = img.size
    draw = ImageDraw.Draw(img)
    
    x1, y1, x2, y2 = clamp_box_xyxy(bbox_xyxy, W, H)
    if not SETTINGS.no_bounding_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)

    if label is not None:
        try:
            font_size = int(img.width // 30) if SETTINGS.font_size == "proportional" else int(SETTINGS.font_size)
            font = ImageFont.truetype(SETTINGS.font_file, font_size)
        except Exception:
            font = ImageFont.load_default()
        tw = draw.textlength(label, font=font)
        th = (font.size + 4) if hasattr(font, "size") else 14

        # clamp X and Y so box fits inside image
        tx = int(x1)
        if tx + tw + 6 > W:
            tx = max(0, W - int(tw) - 6)

        ty = int(y1) - th - 2
        if ty < 0:
            ty = 0

        draw.rectangle([tx, ty, tx + tw + 6, ty + th], fill=(0, 0, 0))
        draw.text((tx + 2, ty + 2), label, fill=color, font=font)

    if segs:
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay, "RGBA")
        for seg in segs:
            if isinstance(seg, list):
                pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                odraw.polygon(pts, fill=color + (ALPHA30,))
                # Close the outline by going back to the first point.
                pts.append((seg[0], seg[1]))
                odraw.line(pts, fill=color + (255,), width=stroke)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return img



def parse_arguments():
    """
    Parse CLI arguments and populate global ``SETTINGS``.

    :returns: None (sets global ``SETTINGS``).
    :rtype: None
    """
    ap = argparse.ArgumentParser(description="Render HFinder predictions.")
    for short, param in ARGLIST.items():
        config = param["config"]
        if "default" in config:
            config["help"] = f"{config['help']} (default: {config['default']})"
        ap.add_argument(short, param["long"], **config)
    global SETTINGS 
    SETTINGS = ap.parse_args()
    print(f"SETTINGS {'-' * 71}")
    for k, v in vars(SETTINGS).items():
        print(f"[INFO] '{k}' = {v}")
    print('-' * 80)



def extract_frame(tif, ch=0, z=0):
    """
    Return a 2D plane (H, W) from an array shaped (C, H, W) or (Z, C, H, W).

    Selection rules:
      - (C, H, W): select channel ``ch`` (default 0).
      - (Z, C, H, W): select slice ``z`` (default 0) and channel ``ch`` (default 0).

    :param tif: Image array.
    :type tif: numpy.ndarray
    :param ch: Channel index to select (defaults to 0 if None).
    :type ch: int | None
    :param z: Z index to select (only used when Z is present; defaults to 0 if None).
    :type z: int | None
    :returns: 2D image plane as (H, W).
    :rtype: numpy.ndarray
    :raises ValueError: If the array does not match (C,H,W) or (Z,C,H,W), or if indices are out of bounds.
    """

    if tif.ndim == 2:
        return tif

    if tif.ndim == 3:
        C, H, W = tif.shape
        if not (0 <= ch < C):
            raise ValueError(f"Channel index {ch} out of range [0, {C-1}] for shape {tif.shape}.")
        return tif[ch, :, :]

    if tif.ndim == 4:
        Z, C, H, W = tif.shape
        if not (0 <= z < Z):
            raise ValueError(f"Z index {z} out of range [0, {Z-1}] for shape {tif.shape}.")
        if not (0 <= ch < C):
            raise ValueError(f"Channel index {ch} out of range [0, {C-1}] for shape {tif.shape}.")
        return tif[z, ch, :, :]

    raise ValueError(
        f"Unsupported image shape {tif.shape}; expected (C,H,W) or (Z,C,H,W)."
    )



def main():
    """
    Entry point: iterate over TIFFs and render one PNG per (category, channel).

    Steps:
      1. Parse arguments and create output folder.
      2. For each ``*.tif[f]``:
         a. Load all matching COCO JSONs and merge annotations.
         b. Extract a 2D plane: first plane by default; per-channel plane when rendering.
         c. For each (category, channel) group:
            - draw boxes/polygons/labels with confidence-coded color if a palette is set,
            - save as ``{basename}_{sanitized-category}_ch{channel}.png``.

    :returns: None
    :rtype: None
    """
    print("HFinder auxiliary script json2image.py")
    parse_arguments()

    Path(SETTINGS.out_dir).mkdir(parents=True, exist_ok=True)
    
    cmap = None
    if SETTINGS.palette is not None:
        try:
            cmap = matplotlib.colormaps.get_cmap(SETTINGS.palette)
            norm = Normalize(vmin=0.0, vmax=1.0)
        except Exception:
            print(f"⚠️ Unknown palette '{SETTINGS.palette}', falling back to cyan.")

    tiff_paths = sorted(
        chain(
            Path(SETTINGS.tiff_dir).glob("*.tif"),
            Path(SETTINGS.tiff_dir).glob("*.tiff")
        )
    )

    for tif_path in tiff_paths:
        print(f"Processing file '{tif_path.name}'")
        base = tif_path.stem
        anns, id_to_name = load_all_coco_for_base(base, SETTINGS.annotations)
        if not anns:
            print("   ❌ No annotations, skipping")
            continue

        # TIFF → background grayscale (2D plane or first plane)
        tif = tifffile.imread(tif_path)
        H, W = tif.shape[-2:]
        stroke = max(1, W // 500)

        by_cat_ch = defaultdict(list)
        for a in anns:
            cid = int(a["category_id"])
            # FIXME: HFinder should not save 'hf_channels' as a list
            raw_ch = a.get("hf_channels", 0)
            if isinstance(raw_ch, list) and raw_ch:
                ch = int(raw_ch[0])
            else:
                ch = int(raw_ch)
            by_cat_ch[cid, ch].append(a)

        planes = {}
        for (cid, ch), cat_anns in sorted(by_cat_ch.items()):
            cls_name = id_to_name.get(cid, f"class_{cid}")
            if ch not in planes:
                frame = normalize_to_uint8(extract_frame(tif, ch=ch))
                planes[ch] = ImageOps.grayscale(Image.fromarray(frame)).convert("RGB")
            canvas = planes[ch].copy()

            for a in cat_anns:
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                conf = float(a.get("confidence", 0.0))
                if SETTINGS.no_labels:
                    label = None
                elif SETTINGS.long_labels:
                    label = f"{cls_name.title()} ({conf:.2f})"
                else:
                    label = f"{cls_name[:2].title()}. ({conf:.2f})"
                canvas = draw_annotation(
                    canvas,
                    bbox_xyxy=[x, y, x + w, y + h],
                    segs=[seg for seg in a.get("segmentation", []) if isinstance(seg, list)],
                    label=label,
                    color=CYAN if cmap is None else tuple(int(255*v) for v in cmap(norm(conf))[:3]),
                    stroke=stroke
                )

            out_name = f"{base}_{sanitize(cls_name)}_ch{ch}.png"
            canvas.save(Path(SETTINGS.out_dir) / out_name)
            print(f"   ✅ Saving '{out_name}'")



if __name__ == "__main__":
    main()
