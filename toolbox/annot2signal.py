#!/usr/bin/env python3
"""
annot2signal.py

This script extracts per-polygon signal intensities from TIFF images using YOLO/COCO-style annotations. 
For each object polygon, it computes the centroid coordinates and mean signal intensity, 
optionally from a different channel than the detection channel (via --signal). 
The results are written to a TSV table, with one row per object.

Outputs:
    - A TSV file with columns: Filename, Class, X, Y, Mean
    - Verification PNG masks where polygons are filled with their original intensities

Usage example:
    python annot2signal.py -t ./tiffs -a ./jsons -o ./out -c Nucleus -s 2
"""

import os
import re
import json
import numpy as np
import tifffile
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
from collections import defaultdict
from PIL import Image, ImageDraw

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
            "help": "Output directory for COCO JSON files"
        }
    },
    "-o": {
        "long": "--output_dir",
        "config": {
            "default": ".",
            "help": "Output directory for PNG files"
        }
    },
    "-cat": {
        "long": "--category",
        "config": {
            "required": True,
            "help": "Category to analyse"
        }
    },
    "-sig": {
        "long": "--signal",
        "config": {
            "default": "same",
            "help": "Index of the channel used to retrieve signal. 'same' = use the detection channel"
        }
    }
}



def sanitize(name):
    """
    Sanitize a string for safe use as a filename.

    :param name: Raw input string
    :type name: str
    :return: Sanitized string with only alphanumerics, dot, dash, underscore
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9+._-]+", "_", name)



def load_coco_json(base, category, coco_dir):
    """
    Load and merge COCO JSON files matching basename and category.

    :param base: Base filename of the TIFF (without extension)
    :type base: str
    :param category: Target category name or ID
    :type category: str
    :param coco_dir: Path to the folder containing JSON files
    :type coco_dir: str | Path
    :return: Tuple of (list of annotations, {category_id: category_name})
    :rtype: tuple[list[dict], dict[int, str]]
    """
    files = sorted(Path(coco_dir).glob(f"{base}_{category}.json"))
    anns, id_to_name = [], {}
 
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)
  
        # Use 0-based hf_channels
        if "hf_channels" in d:
            d["hf_channels"] = [c - 1 for c in d["hf_channels"]]
  
        for c in d.get("categories", []):
            cid = int(c["id"])
            id_to_name[cid] = c.get("name", f"class_{cid}")
        anns.extend(d.get("annotations", []))

    return anns, id_to_name



def extract_frame(arr, ch = 0, z = 0):
    """
    Extract a 2D frame (H,W) from TIFF data.

    :param arr: TIFF array with shape (C,H,W) or (Z,C,H,W)
    :type arr: np.ndarray
    :param ch: Channel index (0-based)
    :type ch: int
    :param z: Z-slice index if 4D
    :type z: int
    :return: 2D image frame
    :rtype: np.ndarray
    :raises ValueError: If channel or z index are out of range
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        C, H, W = arr.shape
        if not (0 <= ch < C):
            raise ValueError(f"Channel {ch} out of range [0,{C-1}] for shape {arr.shape}")
        return arr[ch, :, :]
    if arr.ndim == 4:
        Z, C, H, W = arr.shape
        if not (0 <= z < Z):
            raise ValueError(f"Z {z} out of range [0,{Z-1}] for shape {arr.shape}")
        if not (0 <= ch < C):
            raise ValueError(f"Channel {ch} out of range [0,{C-1}] for shape {arr.shape}")
        return arr[z, ch, :, :]
    raise ValueError(f"Unsupported image shape {arr.shape}; expected (C,H,W) or (Z,C,H,W).")



def polygon_to_mask(polygon, shape_hw):
    """
    Convert a YOLO polygon into a binary mask.

    :param polygon: Polygon vertices [x0,y0,...] or Nx2 array
    :type polygon: list[float] | np.ndarray
    :param shape_hw: Shape of the output mask (H,W)
    :type shape_hw: tuple[int,int]
    :return: Binary mask with True inside polygon
    :rtype: np.ndarray of bool
    """
    H, W = shape_hw

    # Normalize input polygon format
    if isinstance(polygon, list):
        polygon = np.array(polygon).reshape(-1, 2)
    elif isinstance(polygon, np.ndarray) and polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)

    # Draw mask
    mask = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    pts = [(int(x), int(y)) for x, y in polygon]
    draw.polygon(pts, outline=1, fill=1)

    return np.array(mask, dtype=bool)



def measure_polygon_mean(frame, polygon):
    """
    Compute the mean intensity of pixels inside a polygon.

    :param frame: 2D image array
    :type frame: np.ndarray
    :param polygon: Polygon coordinates (flat list [x,y,...] or Nx2 array)
    :type polygon: list[float] | np.ndarray
    :return: Mean intensity of pixels inside the polygon (NaN if empty)
    :rtype: float
    """
    H, W = frame.shape

    if isinstance(polygon, list):
        polygon = np.array(polygon).reshape(-1, 2)
    elif isinstance(polygon, np.ndarray) and polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)

    m = polygon_to_mask(polygon, (H, W))

    if not m.any():
        return float("nan")

    vals = frame[m]
    return float(vals.mean())



def parse_arguments():
    """
    Parse CLI arguments and populate global SETTINGS.

    :return: None (sets global SETTINGS object)
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
    
    # 0-based channel
    if SETTINGS.signal != "same":
        SETTINGS.signal = int(SETTINGS.signal) - 1

    print(f"SETTINGS {'-' * 71}")
    for k, v in vars(SETTINGS).items():
        print(f"[INFO] '{k}' = {v}")
    print('-' * 80)



def main():
    """
    Main entry point.
    Iterates over TIFFs, loads JSON annotations, extracts polygon signals,
    saves verification masks and writes results to TSV.

    :return: None
    :rtype: None
    """
    parse_arguments()
    os.makedirs(SETTINGS.output_dir, exist_ok=True)

    tiff_paths = sorted(
        chain(
            Path(SETTINGS.tiff_dir).glob("*.tif"),
            Path(SETTINGS.tiff_dir).glob("*.tiff")
        )
    )

    cat_tag = sanitize(str(SETTINGS.category))
    out_tsv = Path(SETTINGS.output_dir) / f"values_{cat_tag}.tsv"
    with open(out_tsv, "w") as f:
        f.write("Filename\tClass\tX\tY\tMean\n")
        for tif_path in tiff_paths:
            base = tif_path.stem
            print(f"Processing {tif_path.name}")
            anns, id_to_name = load_coco_json(base, SETTINGS.category, SETTINGS.annotations)
            if not anns:
                print(f"   ⚠️ No annotations, skipping.")
                continue

            # Get class ID
            selected_class = set()
            if SETTINGS.category.isdigit():
                selected_class.add(int(SETTINGS.category))
            else:
                for cid, nm in id_to_name.items():
                    if nm == SETTINGS.category:
                        selected_class.add(cid)
            if not selected_class:
                print(f"   ⚠️ Unknown category {SETTINGS.category}.")
                continue

            arr = tifffile.imread(tif_path)
            H, W = arr.shape[-2:]

            # Sort annotations by channel
            by_ch = defaultdict(list)
            for a in anns:
                cid = int(a["category_id"])
                if cid not in selected_class:
                    continue
                raw_ch = a.get("hf_channels", 0)
                ch = int(raw_ch[0]) if (isinstance(raw_ch, list) and raw_ch) else int(raw_ch)
                segs = [seg for seg in a.get("segmentation", []) if isinstance(seg, list) and len(seg) >= 6]
                if segs:
                    by_ch[ch].append(segs)

            if not by_ch:
                print(f"   ⚠️ No annotations, skipping.")
                continue

            for ch, poly_list in sorted(by_ch.items()):
                if SETTINGS.signal == "same":
                    signal_ch = ch
                else:
                    signal_ch = int(SETTINGS.signal)
                    if not (0 <= signal_ch < arr.shape[-3 if arr.ndim == 4 else -3+1]):
                        raise ValueError(f"   ❌ Signal channel {signal_ch} out of bounds for shape {arr.shape}")
                frame = extract_frame(arr, ch=signal_ch)
           
                #union_mask = np.zeros((H, W), dtype=bool)
                for i, segs in enumerate(poly_list):
                    mask = polygon_to_mask(segs, (H, W))
                    # Image restreinte au polygone
                    masked_frame = np.zeros_like(frame)
                    masked_frame[mask] = frame[mask]
                    out_png = Path(SETTINGS.output_dir) / f"{base}_{i}_mask.png"
                    plt.imsave(out_png, masked_frame, cmap="gray")
                    #union_mask |= polygon_to_mask(segs, (H, W))
                    seg_mean = measure_polygon_mean(frame, segs)
                    
                    M = mask.sum()
                    if M > 0:
                        ys, xs = np.nonzero(mask)
                        cx, cy = xs.mean(), H - 1 - ys.mean()
                    else:
                        cx, cy = np.nan, np.nan
                    
                    f.write(f"{tif_path.name}\t{SETTINGS.category}\t{cx}\t{cy}\t{seg_mean}\n")

    print(f"✅ Data saved in file '{out_tsv}'")


if __name__ == "__main__":
    main()

