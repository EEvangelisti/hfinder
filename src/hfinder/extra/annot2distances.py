"""
Compute distances between annotated structures in confocal TIFF images and
export visualization overlays with a hue-based composite that avoids saturation.

**What it does**
- Loads TIFFs (shapes: (C,H,W) or (Z,C,H,W)) and COCO/YOLO-style JSON annotations (polygons).
- Extracts per-instance centroids (geometric) for chosen classes.
- Computes distances:
  - A↔B (nearest-neighbor from each A to the closest B), or
  - within-class (nearest-neighbor A↔A) if only one class is given.
- Saves a tidy CSV of distances and an RGB composite overlay using HSV fusion
  (distinct hues per channel; vivid, non-dull, non-saturating).

**Usage (examples)**
.. code-block:: bash

    # Distances from class "Nucleus" to class "Haustorium" on Z=0, channel hues 0,120,240 deg
    python json2distances.py -d ./tiffs -c ./jsons -o ./out \
        --classA Nucleus --classB Haustorium --z 0 --hues 0,120,240

    # Within-class distances for "Nucleus", composite using default hues
    python json2distances.py -d ./tiffs -c ./jsons -o ./out \
        --classA Nucleus --z 0

Outputs:
- distances.csv — per-image, per-instance nearest-neighbor distances (pixels).
- overlay_<basename>.png — HSV composite overlay with polygons/centroids drawn.

Notes:
- Polygons are expected to be single YOLO-style polygons per instance.
- Channels are treated 0-based internally. If your metadata is 1-based, convert at parse time.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from itertools import chain
from typing import Iterable, Tuple, List, Dict, Optional

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from hfinder.core import utils as HF_utils

SETTINGS = None
ARGLIST = HF_utils.load_argument_list("annot2distances.arglist.json") or {}



# ----------------------------- I/O & parsing ----------------------------- #

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
        if "type" in config:
            config["type"] = HF_utils.string_to_typefun(config["type"])
        ap.add_argument(short, param["long"], **config)
    global SETTINGS 
    SETTINGS = ap.parse_args()
    print(f"SETTINGS {'-' * 71}")
    for k, v in vars(SETTINGS).items():
        print(f"[INFO] '{k}' = {v}")
    print('-' * 80)


def load_all_coco_for_base(base: str, coco_dir: str | Path) -> tuple[list[dict], dict[int, str]]:
    """
    Load and merge COCO-like JSONs whose basename starts with ``base``.

    :param base: TIFF basename (without extension).
    :type base: str
    :param coco_dir: Folder with JSON files.
    :type coco_dir: str | Path
    :return: (annotations, id_to_name) where id_to_name maps numeric id → class name.
    :rtype: tuple[list[dict], dict[int, str]]
    """
    files = sorted(Path(coco_dir).glob(f"{base}*.json"))
    anns: List[dict] = []
    id_to_name: Dict[int, str] = {}
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)
        for c in d.get("categories", []):
            cid = int(c["id"])
            id_to_name[cid] = c.get("name", f"class_{cid}")
        anns.extend(d.get("annotations", []))
    return anns, id_to_name


# ----------------------------- Image helpers ----------------------------- #

def extract_frame(arr: np.ndarray, ch: int = 0, z: int = 0) -> np.ndarray:
    """
    Return a 2D plane (H,W) from an array shaped (C,H,W) or (Z,C,H,W).

    :param arr: Image array.
    :type arr: np.ndarray
    :param ch: Channel index (0-based).
    :type ch: int
    :param z: Z index if present.
    :type z: int
    :return: 2D frame.
    :rtype: np.ndarray
    :raises ValueError: If shape unsupported or indices out of range.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        C, H, W = arr.shape
        if not (0 <= ch < C):
            raise ValueError(f"Channel {ch} out of range for shape {arr.shape}")
        return arr[ch]
    if arr.ndim == 4:
        Z, C, H, W = arr.shape
        if not (0 <= z < Z):
            raise ValueError(f"Z {z} out of range for shape {arr.shape}")
        if not (0 <= ch < C):
            raise ValueError(f"Channel {ch} out of range for shape {arr.shape}")
        return arr[z, ch]
    raise ValueError(f"Unsupported image shape {arr.shape}; expected (C,H,W) or (Z,C,H,W).")


def normalize_to_unit(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a frame to [0,1] as float32 (robust min–max).

    :param frame: 2D array (any dtype).
    :type frame: np.ndarray
    :return: Float frame in [0,1].
    :rtype: np.ndarray
    """
    a = frame.astype(np.float32, copy=False)
    vmin, vmax = float(np.min(a)), float(np.max(a))
    if vmax <= vmin:
        return np.zeros_like(a, dtype=np.float32)
    return (a - vmin) / (vmax - vmin)


def hsv_composite_from_channels(arr: np.ndarray,
                                hues_deg: list[float] | tuple[float, ...] | np.ndarray,
                                z: int = 0,
                                channels: list[int] | tuple[int, ...] | np.ndarray | None = None
                                ) -> Image.Image:
    """
    Build a vivid HSV composite from selected channels.

    :param arr: TIFF array of shape (C,H,W) or (Z,C,H,W) or (H,W).
    :type arr: np.ndarray
    :param hues_deg: Hues in degrees [0..360] to map *selected* channels.
                     Length will be matched to the number of selected planes.
    :type hues_deg: list[float] | tuple[float, ...] | np.ndarray
    :param z: Z index if arr is 4D.
    :type z: int
    :param channels: Channel indices to include (0-based). If None, include all.
    :type channels: list[int] | tuple[int, ...] | np.ndarray | None
    :return: RGB composite image (PIL).
    :rtype: PIL.Image.Image
    """
    # Normalize shapes and choose indices to use
    if arr.ndim == 2:
        planes_src = [arr]
        idx = np.array([0], dtype=int)
    elif arr.ndim == 3:
        C = arr.shape[0]
        idx = np.arange(C, dtype=int) if channels is None else np.array(channels, dtype=int)
        planes_src = [arr[c] for c in idx]
    elif arr.ndim == 4:
        Z, C = arr.shape[0], arr.shape[1]
        if not (0 <= z < Z):
            raise ValueError(f"Z index {z} out of range for shape {arr.shape}")
        idx = np.arange(C, dtype=int) if channels is None else np.array(channels, dtype=int)
        planes_src = [arr[z, c] for c in idx]
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}; expected (H,W), (C,H,W) or (Z,C,H,W).")

    if len(planes_src) == 0:
        raise ValueError("No channels selected for HSV composite (empty 'channels').")

    # Ensure each plane is 2D and normalized to [0,1]
    planes = []
    for k, p in enumerate(planes_src):
        if p.ndim != 2:
            raise ValueError(f"Selected plane at index {k} is not 2D: got shape {p.shape}")
        planes.append(normalize_to_unit(p))

    H, W = planes[0].shape
    Csel = len(planes)

    # Prepare hues for the selected channels
    hues_deg = list(hues_deg) if not isinstance(hues_deg, list) else hues_deg
    if len(hues_deg) < Csel:
        hues_deg = hues_deg + [0.0] * (Csel - len(hues_deg))  # pad
    hues = np.asarray([(h % 360) / 360.0 for h in hues_deg[:Csel]], dtype=np.float32)

    # Stack and compute HSV fields
    stack = np.stack(planes, axis=0)  # (Csel,H,W)
    vmax = np.max(stack, axis=0)      # (H,W)
    argm = np.argmax(stack, axis=0)   # (H,W)
    nonzero = (stack > 0.0).sum(axis=0).astype(np.float32)
    s_approx = np.clip(stack.sum(axis=0) / (1e-6 + np.maximum(1.0, nonzero)), 0.0, 1.0)

    H_hsv = hues[argm]   # (H,W)
    S_hsv = s_approx     # (H,W)
    V_hsv = vmax         # (H,W)

    rgb = hsv_to_rgb_np(H_hsv, S_hsv, V_hsv)  # (H,W,3) in [0,1]
    return Image.fromarray((rgb * 255.0).astype(np.uint8)).convert("RGB")



def hsv_to_rgb_np(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV→RGB (all arrays same shape), returning float RGB in [0,1].

    :param h: Hue in [0,1].
    :type h: np.ndarray
    :param s: Saturation in [0,1].
    :type s: np.ndarray
    :param v: Value in [0,1].
    :type v: np.ndarray
    :return: RGB array shape (H,W,3) float in [0,1].
    :rtype: np.ndarray
    """
    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = i % 6

    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# ----------------------------- Geometry & masks ----------------------------- #

def polygon_to_mask(polygon: list[float] | np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """
    Rasterize a single YOLO-style polygon to a boolean mask.

    :param polygon: Flat list [x0,y0,...] or Nx2 array of vertices.
    :type polygon: list[float] | np.ndarray
    :param shape_hw: Target mask shape (H,W).
    :type shape_hw: tuple[int,int]
    :return: Mask where True indicates pixels inside the polygon.
    :rtype: np.ndarray
    """
    H, W = shape_hw
    if isinstance(polygon, list):
        poly = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    elif isinstance(polygon, np.ndarray) and polygon.ndim == 1:
        poly = polygon.reshape(-1, 2).astype(np.float32)
    else:
        poly = polygon.astype(np.float32)

    img = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(img)
    pts = [(int(x), int(y)) for x, y in poly]
    draw.polygon(pts, outline=1, fill=1)
    return np.array(img, dtype=bool)


def polygon_centroid_xy(mask: np.ndarray) -> tuple[float, float]:
    """
    Compute geometric centroid (mean of pixel coordinates) from a boolean mask.

    :param mask: Boolean mask of the polygon.
    :type mask: np.ndarray
    :return: (cx, cy) with origin at top-left (image coordinates).
    :rtype: tuple[float, float]
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


# ----------------------------- Distances ----------------------------- #



def pairwise_nndists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute nearest-neighbor distance from each point in A to the closest point in B.

    :param A: Array of shape (NA, 2) with (x,y) coordinates.
    :type A: np.ndarray
    :param B: Array of shape (NB, 2) with (x,y) coordinates.
    :type B: np.ndarray
    :return: Distances array of shape (NA,), NaN if B is empty.
    :rtype: np.ndarray
    """
    if A.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if B.size == 0:
        return np.full((A.shape[0],), np.nan, dtype=np.float32)
    # Efficient squared distances via broadcasting (safe for moderate N)
    d2 = (A[:, None, 0] - B[None, :, 0]) ** 2 + (A[:, None, 1] - B[None, :, 1]) ** 2
    return np.sqrt(np.min(d2, axis=1)).astype(np.float32)


# ----------------------------- Drawing overlay ----------------------------- #

def draw_polygons_and_centroids(rgb: Image.Image,
                                instances: list[tuple[list[float], str]],
                                line: int = 2,
                                font_path: str = "DejaVuSans.ttf",
                                font_size: int = 14,
                                connect: tuple[str, str] | None = None) -> Image.Image:
    """
    Draw polygon outlines and centroids with small labels on top of an RGB image.
    Optionally draw lines between centroids of two categories.

    :param rgb: Background RGB image.
    :type rgb: PIL.Image.Image
    :param instances: List of (polygon, label) tuples.
    :type instances: list[tuple[list[float], str]]
    :param line: Stroke width.
    :type line: int
    :param font_path: TTF font path.
    :type font_path: str
    :param font_size: Font size.
    :type font_size: int
    :param connect: Pair of class labels (labelA, labelB) to connect centroids.
    :type connect: tuple[str, str] | None
    :return: Image with overlays.
    :rtype: PIL.Image.Image
    """
    im = rgb.convert("RGBA")
    odraw = ImageDraw.Draw(im, "RGBA")

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    centroids = []
    for poly, label in instances:
        # outline
        if isinstance(poly, list):
            pts = [(int(poly[i]), int(poly[i+1])) for i in range(0, len(poly), 2)]
        else:
            pts = [(int(x), int(y)) for x, y in poly.reshape(-1, 2)]
        odraw.line(pts + [pts[0]], fill=(255, 255, 255, 220), width=line)

        # centroid + label
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        r = max(2, line)
        odraw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 220))
        if label and False: # FIXME
            odraw.text((cx + 3*r, cy), label, fill=(255, 255, 255, 220), font=font)

        centroids.append(((cx, cy), label))

    # --- optional connecting lines ---
    if connect is not None:
        labelA, labelB = connect
        ptsA = [c for c, l in centroids if l == labelA]
        targetB = labelA if labelB is None else labelB
        ptsB = [c for c, l in centroids if l == targetB]

        for (ax, ay) in ptsA:
            for (bx, by) in ptsB:
                odraw.line([(ax, ay), (bx, by)], fill=(255, 255, 255, 120), width=line)

    return im.convert("RGB")


# ----------------------------- Main ----------------------------- #

def main():
    """
    CLI entry-point: compute distances and save overlays/CSV.

    :return: None
    :rtype: None
    """
    parse_arguments()
    out_dir = Path(SETTINGS.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Parse hues (degrees) for channels → list[float]
    hues_deg = [float(x.strip()) for x in SETTINGS.hues_list.split(",") if x.strip()]

    rows: List[tuple] = []
    saved_png = 0

    tiff_paths = sorted(chain(Path(SETTINGS.tiff_dir).glob("*.tif"),
                              Path(SETTINGS.tiff_dir).glob("*.tiff")))
    for tif_path in tiff_paths:
        base = tif_path.stem
        anns, id2name = load_all_coco_for_base(base, SETTINGS.annotations)
        if not anns:
            continue

        # Resolve classes (accept numeric id or exact name)
        def normalize_class(token: str) -> set[int]:
            if token is None:
                return set()
            if token.isdigit():
                return {int(token)}
            return {cid for cid, nm in id2name.items() if nm == token}

        A_ids = normalize_class(SETTINGS.class_A)
        B_ids = normalize_class(SETTINGS.class_B) if SETTINGS.class_B else set()

        arr = tifffile.imread(tif_path)
        # Build composite (HSV) for visualization
        channels_used = set()
        for ann in anns:
            chs = ann.get("hf_channels", [])
            if isinstance(chs, int):
                channels_used.add(chs)  # COCO stores 1-based, numpy is 0-based
            elif isinstance(chs, list):
                for ch in chs:
                    channels_used.add(ch)
        channels_used = np.array(sorted(channels_used), dtype=int)
        # FIXME: create an option for Z value when processing Z-stacks.
        composite = hsv_composite_from_channels(arr, hues_deg, z=0, channels=channels_used)

        # Collect polygons by class
        polys_A: List[list[float]] = []
        polys_B: List[list[float]] = []

        for a in anns:
            cid = int(a["category_id"])
            segs = a.get("segmentation", [])
            if not segs:
                continue
            poly = segs[0] if isinstance(segs, list) else segs  # YOLO: single polygon
            if cid in A_ids:
                polys_A.append(poly)
            elif SETTINGS.class_B and cid in B_ids:
                polys_B.append(poly)

        # Centroids
        H = composite.height; W = composite.width
        A_pts = []
        for poly in polys_A:
            m = polygon_to_mask(poly, (H, W))
            cx, cy = polygon_centroid_xy(m)
            if np.isfinite(cx) and np.isfinite(cy):
                A_pts.append((cx, cy))
        A_pts = np.asarray(A_pts, dtype=np.float32).reshape(-1, 2)

        if SETTINGS.class_B:
            B_pts = []
            for poly in polys_B:
                m = polygon_to_mask(poly, (H, W))
                cx, cy = polygon_centroid_xy(m)
                if np.isfinite(cx) and np.isfinite(cy):
                    B_pts.append((cx, cy))
            B_pts = np.asarray(B_pts, dtype=np.float32).reshape(-1, 2)

            for i, (ax, ay) in enumerate(A_pts):
                for j, (bx, by) in enumerate(B_pts):
                    d = np.hypot(ax - bx, ay - by)  # distance euclidienne
                    rows.append((base, "A->B", f"{i}->{j}", d))
        else:
            # Within-class NND for A
            if A_pts.shape[0] >= 2:
                for i in range(len(A_pts)):
                    for j in range(i+1, len(A_pts)):  # éviter doublons
                        d = np.hypot(A_pts[i,0] - A_pts[j,0], A_pts[i,1] - A_pts[j,1])
                        rows.append((base, "A->A", f"{i}<->{j}", d))
            else:
                # not enough points for within-class NND
                pass

        # Draw overlay with polygon outlines + centroids (labels are class names)
        if saved_png < SETTINGS.max_png:
            lab_instances = []
            for poly in polys_A:
                lab_instances.append((poly, f"{SETTINGS.class_A}"))
            if SETTINGS.class_B:
                for poly in polys_B:
                    lab_instances.append((poly, f"{SETTINGS.class_B}"))

            vis = draw_polygons_and_centroids(composite, lab_instances,
                                              line=SETTINGS.stroke,
                                              font_path=SETTINGS.font_file,
                                              font_size=SETTINGS.font_size,
                                              connect=(SETTINGS.class_A, SETTINGS.class_B))
            out_png = out_dir / f"overlay_{base}.png"
            vis.save(out_png)
            saved_png += 1

    # Save CSV
    if rows:
        out_csv = out_dir / "distances.csv"
        with open(out_csv, "w") as f:
            f.write("Image,Pair,i,DistancePx\n")
            for base, pair, i, d in rows:
                f.write(f"{base},{pair},{i},{d}\n")

    return 0


