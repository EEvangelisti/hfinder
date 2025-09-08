#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
        [--palette viridis] [--labels] [--boxes] \
        [--font_file DejaVuSans.ttf] [--font_size 18]

"""

import re
import ast
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


DEFAULT_COLOR = (0, 255, 255)
ALPHA30 = int(0.30 * 255)
SETTINGS = None

ARGLIST = {
    "-a": {
        "long": "--annotations",
        "config": {
            "default": ".",
            "help": "Folder containing COCO JSON files"
        }
    },
    "-box": {
        "long": "--boxes",
        "config": {
            "action": "store_true",
            "help": "Display bounding boxes around polygons"
        }
    },
    "-cat": {
        "long": "--category",
        "config": {
            "default": "*",
            "help": "Process the given category only"
        }
    },
    "-cmp": {
        "long": "--composite",
        "config": {
            "action": "store_true",
            "help": "Générer une image composite unique avec tous les polygones"
        }
    },
    "-cmpbg": {
        "long": "--composite_bg",
        "config": {
            "default": "black",
            "help": "Fond de l'image composite: 'black', 'avg' (moyenne des canaux) ou 'max' (maximum par pixel)"
        }
    },
    "-cmplab": {
        "long": "--composite_labels",
        "config": {
            "action": "store_true",
            "help": "Afficher les labels sur l'image composite (par défaut: masqués)"
        }
    },
    "-cmpbox": {
        "long": "--composite_boxes",
        "config": {
            "action": "store_true",
            "help": "Afficher les boîtes englobantes sur l'image composite (par défaut: masquées)"
        }
    },
    "-t": {
        "long": "--tiff_dir",
        "config": {
            "default": ".",
            "help": "Folder containing TIFF files"
        }
    },
    "-lab": {
        "long": "--labels",
        "config": {
            "action": "store_true",
            "help": "Display labels and confidence values"
        }
    },
    "-long": {
        "long": "--long_labels",
        "config": {
            "action": "store_true",
            "help": "Do not abbreviate label names"
        }
    },
    "-o": {
        "long": "--out_dir",
        "config": {
            "default": ".",
            "help": "Output directory for PNG files"
        }
    },
    "-pal": {
        "long": "--palette",
        "config": {
            "default": None,
            "help": "Matplotlib colormap used to encode confidence values (e.g. viridis, plasma, cool) or #RRGGBB value"
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
    }
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



import math
from PIL import Image, ImageDraw, ImageFont

ALPHA30 = 77  # ≈30% d'opacité

def _split_poly_by_jumps(seg, max_jump=10.0):
    """Découpe un seg (liste [x0,y0,x1,y1,...]) en sous-polygones
    en cassant aux sauts > max_jump pixels."""
    pts = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
    if len(pts) < 3:
        return []
    polys, cur = [], [pts[0]]
    for a, b in zip(pts, pts[1:]):
        (x0, y0), (x1, y1) = a, b
        if math.hypot(x1 - x0, y1 - y0) > max_jump and len(cur) >= 3:
            polys.append(cur)
            cur = [b]
        else:
            cur.append(b)
    if len(cur) >= 3:
        polys.append(cur)
    return polys



def draw_annotation(img, bbox_xyxy, segs, label, color, stroke, composite_mode=False):
    """
    Draw a bounding box, optional filled polygons (30% alpha), and a label on an image.

    - Bounding box is skipped if ``SETTINGS.boxes`` is False.
    - Label is drawn on a black rectangle that clamps to image bounds.
    - Polygons are composited on an RGBA overlay to preserve alpha.

    :param img: PIL image (RGB).
    :type img: PIL.Image.Image
    :param bbox_xyxy: Bounding box as ``[x1, y1, x2, y2]``, or None.
    :type bbox_xyxy: list | tuple | None
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
    
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = clamp_box_xyxy(bbox_xyxy, W, H)
        if (composite_mode and SETTINGS.composite_boxes) or SETTINGS.boxes:
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

    # --- Polygones (sans traits parasites) ---
    if segs:
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay, "RGBA")

        for seg in segs:
            if not isinstance(seg, list) or len(seg) < 6:
                continue  # pas assez de points
            # points (x,y) clampés dans l’image
            pts = [(max(0, min(W - 1, seg[i])),
                    max(0, min(H - 1, seg[i+1])))
                   for i in range(0, len(seg), 2)]
            # Remplissage + contour en une seule primitive : pas de "line" manuelle
            try:
                odraw.polygon(pts,
                              fill=color + (ALPHA30,),
                              outline=color + (255,),
                              width=stroke)
            except TypeError:
                # Pillow ancien sans 'width' : on trace d'abord le remplissage,
                # puis un contour fin via polygon(outline=...) sans 'width'
                odraw.polygon(pts, fill=color + (ALPHA30,))
                odraw.polygon(pts, outline=color + (255,))

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
        if not (0 <= ch // Z < C): # FIXME: is is correct for Z stacks?
            raise ValueError(f"Channel index {ch} out of range [0, {C-1}] for shape {tif.shape}.")
        return tif[z, ch // Z, :, :] # FIXME: is is correct for Z stacks?

    raise ValueError(
        f"Unsupported image shape {tif.shape}; expected (C,H,W) or (Z,C,H,W)."
    )



def hex_to_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))



def to_uint8_rgb(c):
    if isinstance(c, str):
        if c.startswith("#"):
            return hex_to_rgb(c)
        r, g, b = to_rgb(c)  # matplotlib gère noms de couleurs
        return (int(r*255), int(g*255), int(b*255))
    if isinstance(c, (tuple, list)) and len(c) == 3:
        if max(c) > 1.0:
            return tuple(int(v) for v in c)
        return tuple(int(v*255) for v in c)
    raise ValueError(f"Unsupported color format: {c}")



def parse_palette_arg(arg):
    """Try to interpret a palette argument string as dict, hex color, or name."""
    # Tenter un dict
    if arg.strip().startswith("{"):
        try:
            return ast.literal_eval(arg)
        except Exception:
            raise ValueError(f"Invalid dict for palette: {arg}")
    return arg  # sinon, c'est un str (hex ou cmap)



def resolve_palette(palette):
    """
    Resolve palette specification into one of three cases:
      - dict {class -> (r,g,b)} if palette is a dict
      - (r,g,b) uint8 tuple if palette is a single color
      - (cmap, norm) if palette is a colormap name

    :param palette: Palette specification
    :type palette: dict[str,str|tuple] | str | None
    :param default_color: Fallback color if resolution fails
    :type default_color: str
    :rtype: dict | tuple[int,int,int] | (Colormap, Normalize)
    """

    if palette is None:
        return DEFAULT_COLOR

    palette = parse_palette_arg(palette)

    if isinstance(palette, dict):
        out = {}
        for cls, col in palette.items():
            try:
                out[cls] = to_uint8_rgb(col)
            except Exception:
                out[cls] = DEFAULT_COLOR
        return out

    if isinstance(palette, str) and palette.startswith("#"):
        try:
            return to_uint8_rgb(palette)
        except Exception:
            return DEFAULT

    if isinstance(palette, str):
        try:
            cmap = matplotlib.colormaps.get_cmap(palette)
            norm = Normalize(vmin=0.0, vmax=1.0)
            return (cmap, norm)
        except Exception:
            return DEFAULT_COLOR

    return DEFAULT_COLOR


def _used_channel_from_ann(a):
    """Récupère l'index de canal (1-based) tel que stocké par HFinder."""
    ch = int(a.get("hf_channel", 1))
    return max(0, ch - 1)



def build_background_image(tif, used_channels, mode="black"):
    """
    Construit le fond pour la composite:
      - 'black' → fond noir.
      - 'avg'   → moyenne (grayscale) des canaux utilisés.
      - 'max'   → maximum (grayscale) des canaux utilisés.
    """
    H, W = tif.shape[-2:]
    if mode.lower() == "black" or not used_channels:
        return Image.new("RGB", (W, H), (0, 0, 0))

    # Empilement des frames normalisées (uint8)
    frames = []
    for ch in sorted(used_channels):
        try:
            frame = normalize_to_uint8(extract_frame(tif, ch=ch))
            frames.append(frame.astype(np.float32))
        except Exception:
            continue

    if not frames:
        # Repli si rien n'est exploitable: premier canal
        frame = normalize_to_uint8(extract_frame(tif, ch=0))
        return ImageOps.grayscale(Image.fromarray(frame)).convert("RGB")

    stack = np.stack(frames, axis=0)  # [C, H, W]

    mode_l = mode.lower()
    if mode_l == "max":
        bg = np.max(stack, axis=0)
    else:  # 'avg' par défaut
        bg = np.mean(stack, axis=0)

    bg8 = np.clip(bg, 0, 255).astype(np.uint8)
    return ImageOps.grayscale(Image.fromarray(bg8)).convert("RGB")


def generate_all_polygons_image(tif, anns, id_to_name, mask_color,
                                background_mode="black",
                                show_labels=False,
                                show_boxes=False):
    """
    Génère une image unique contenant tous les polygones et, selon l'option,
    un fond noir ou la moyenne en niveaux de gris des canaux utilisés.

    :param tif: tableau TIFF (ndarray)
    :param anns: liste d'annotations COCO fusionnées
    :param id_to_name: dict {category_id -> name}
    :param mask_color: tuple RGB, dict {class->RGB} ou (cmap, norm) (sortie de resolve_palette)
    :param background_mode: 'black' ou 'avg'
    :return: PIL.Image (RGB)
    """
    H, W = tif.shape[-2:]
    used_channels = {_used_channel_from_ann(a) for a in anns}
    canvas = build_background_image(tif, used_channels, mode=background_mode)

    stroke = max(1, W // 500)

    for a in anns:
        # bbox
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            continue

        # label (respecte vos options existantes)
        cid = int(a["category_id"])
        cls_name = id_to_name.get(cid, f"class_{cid}")
        conf = float(a.get("confidence", 0.0))

        # --- Labels : par défaut masqués ; affichés si --composite_labels
        if show_labels:
            if SETTINGS.long_labels:
                label = f"{cls_name.title()} ({conf:.2f})"
            else:
                label = f"{cls_name[:2].title()}. ({conf:.2f})"
        else:
            label = None    

        # couleur (mêmes règles que votre rendu par catégorie)
        if isinstance(mask_color, tuple) and len(mask_color) == 2:
            # (cmap, norm)
            cmap, norm = mask_color
            color = tuple(int(255 * v) for v in cmap(norm(conf))[:3])
        elif isinstance(mask_color, dict):
            color = mask_color.get(cls_name, DEFAULT_COLOR)
        else:
            # tuple RGB direct
            color = mask_color

        # segments
        segs = [seg for seg in a.get("segmentation", []) if isinstance(seg, list)]

        # --- Boîtes : passées uniquement si --composite_boxes
        bbox_xyxy = [x, y, x + w, y + h] if show_boxes else None

        # dessin (réutilise votre routine robuste)
        canvas = draw_annotation(
            canvas,
            bbox_xyxy=bbox_xyxy,
            segs=segs,
            label=label,
            color=color,
            stroke=stroke,
            composite_mode=True
        )

    return canvas


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
    MASK_COLOR = resolve_palette(SETTINGS.palette)

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
            ch = int(a.get("hf_channel", 1))
            by_cat_ch[cid, ch - 1].append(a)

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

                label = None
                if SETTINGS.labels:
                    if SETTINGS.long_labels:
                        label = f"{cls_name.title()} ({conf:.2f})"
                    else:
                        label = f"{cls_name[:2].title()}. ({conf:.2f})"
                
                if isinstance(MASK_COLOR, tuple) and len(MASK_COLOR) == 2:
                    cmap, norm = MASK_COLOR
                    color = tuple(int(255*v) for v in cmap(norm(conf))[:3])
                elif isinstance(MASK_COLOR, dict):
                    color = MASK_COLOR[cls_name] if cls_name in MASK_COLOR else DEFAULT_COLOR
                else:
                    color = MASK_COLOR
                canvas = draw_annotation(
                    canvas,
                    bbox_xyxy=[x, y, x + w, y + h],
                    segs=[seg for seg in a.get("segmentation", []) if isinstance(seg, list)],
                    label=label,
                    color=color,
                    stroke=stroke
                )

            out_name = f"{base}_{sanitize(cls_name)}_ch{ch}.png"
            canvas.save(Path(SETTINGS.out_dir) / out_name)
            print(f"   ✅ Saving '{out_name}'")

        # --- Image composite unique (optionnelle) ---
        if SETTINGS.composite:
            try:
                composite = generate_all_polygons_image(
                    tif=tif,
                    anns=anns,
                    id_to_name=id_to_name,
                    mask_color=MASK_COLOR,
                    background_mode=SETTINGS.composite_bg,
                    show_labels=SETTINGS.composite_labels,
                    show_boxes=SETTINGS.composite_boxes
                )
                out_name = f"{base}_ALL_{SETTINGS.composite_bg}.png"
                composite.save(Path(SETTINGS.out_dir) / out_name)
                print(f"   ✅ Saving composite '{out_name}'")
            except Exception as e:
                print(f"   ⚠️  Composite failed: {e}")


if __name__ == "__main__":
    main()
