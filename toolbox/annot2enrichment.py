#!/usr/bin/env python3
"""
annot2enrichment.py — Quantifier l’enrichissement d’intensité par polygone et tracer un box plot.

Fonctions clés
- Parcourt des TIFF confocaux (formes (C,H,W) ou (Z,C,H,W)).
- Charge les annotations associées (même basename).
- Sélectionne une catégorie (``-cat/--category``).
- Par image et canal :
  1) crée le masque union de tous les polygones de la catégorie ;
  2) détermine un **seuil** d’intensité (Otsu par défaut, ou valeur fournie via ``-th/--threshold``) ;
  3) calcule la **moyenne de fond** sur les pixels **hors polygones ET au-dessus du seuil** ;
  4) calcule la **moyenne d’intensité** pour chaque polygone **au-dessus du seuil** (indépendants) ;
  5) en déduit le **ratio d’enrichissement** = mean(polygone)/mean(fond).
- Produit un **box plot** avec les **points individuels** et (en option) un TSV.

Usage
.. code-block:: bash

    python annot2enrichment.py \
        -d /path/to/tiffs \
        -c /path/to/jsons \
        -o /path/to/out \
        -cat "Noyau" \
        [--z 0] \
        [--threshold 1200]   # sinon Otsu par défaut
        [--save-tsv]
"""

import os
import re
import json
import argparse
from pathlib import Path
from itertools import chain
from collections import defaultdict

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")  # rendu hors écran
import matplotlib.pyplot as plt
from skimage import filters
from PIL import Image, ImageDraw


# ----------------------------- Utilitaires ----------------------------- #

def sanitize(name: str) -> str:
    """Sanitize pour noms de fichiers."""
    return re.sub(r"[^A-Za-z0-9+._-]+", "_", name)


def load_all_coco_for_base(base, category, coco_dir):
    """
    Charge et fusionne les JSON dont le basename commence par ``base``.
    :returns: (annotations, id_to_name)
    """
    files = sorted(Path(coco_dir).glob(f"{base}_{category}.json"))
    anns, id_to_name = [], {}
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)
        for c in d.get("categories", []):
            cid = int(c["id"])
            id_to_name[cid] = c.get("name", f"class_{cid}")
        anns.extend(d.get("annotations", []))
    return anns, id_to_name


def extract_frame(arr: np.ndarray, ch: int = 0, z: int = 0) -> np.ndarray:
    """Retourne un plan 2D (H,W) depuis (C,H,W) ou (Z,C,H,W)."""
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


def polygons_to_mask(polygons: list, shape_hw: tuple[int, int]) -> np.ndarray:
    """
    Rasterise des polygones style COCO en masque booléen.
    polygons: liste de listes plates [x0,y0,x1,y1,...], possiblement multiples par instance.
    """
    H, W = shape_hw
    mask = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for seg in polygons:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        pts = [(int(seg[i]), int(seg[i + 1])) for i in range(0, len(seg), 2)]
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def apply_threshold(img: np.ndarray, method: str = "otsu") -> float:
    """
    Apply a scikit-image thresholding method and return the numeric threshold.

    Supported methods: "otsu", "isodata", "li", "yen", "triangle", "mean"
    """
    method = method.lower()
    if method == "otsu":
        return float(filters.threshold_otsu(img))
    elif method == "isodata":
        return float(filters.threshold_isodata(img))
    elif method == "li":
        return float(filters.threshold_li(img))
    elif method == "yen":
        return float(filters.threshold_yen(img))
    elif method == "triangle":
        return float(filters.threshold_triangle(img))
    elif method == "mean":
        return float(filters.threshold_mean(img))
    else:
        raise ValueError(f"Unknown thresholding method: {method}")


# ------------------------- Mesure & normalisation ------------------------- #

def measure_polygon_means(frame, polygons, thr=0):
    """
    Moyenne d’intensité par polygone (pixels au-dessus du seuil si thr non None).
    """
    H, W = frame.shape
    means = []
    for poly in polygons:
        # normalise : une instance peut être [seg1, seg2, ...] ou directement seg
        segs = poly if (isinstance(poly, list) and poly and isinstance(poly[0], list)) else [poly]
        m = polygons_to_mask(segs, (H, W))
        if thr is not None:
            m = m & (frame >= thr)
        if not m.any():
            means.append(float("nan"))
            continue
        vals = frame[m]
        means.append(float(vals.mean()))
    return means[0]


def background_mean_excluding(frame: np.ndarray, mask_union: np.ndarray, thr: float | None) -> float:
    """
    Moyenne du fond sur les pixels **hors polygones** et **au-dessus du seuil** (si thr non None).
    """
    bg = ~mask_union
    if thr is not None:
        bg = bg & (frame >= thr)
    if not bg.any():
        return float("nan")
    return float(frame[bg].mean())


# ------------------------------- Arguments ------------------------------- #

def parse_arguments():
    ap = argparse.ArgumentParser(description="Extraire l’enrichissement d’intensité par polygone et tracer un box plot.")
    ap.add_argument("-d", "--tiff_dir", default=".", help="Dossier des TIFF (default: .)")
    ap.add_argument("-c", "--coco_dir", default=".", help="Dossier des annotations JSON (default: .)")
    ap.add_argument("-o", "--out_dir",  default=".", help="Dossier de sortie (default: .)")
    ap.add_argument("-cat", "--category", required=True,
                    help="Catégorie à analyser (id numérique ou nom exact).")
    ap.add_argument(
        "-s", "--signal",
        default="same",
        help="Index of the channel used to retrieve signal. 'same' = use the detection channel."
    )
    ap.add_argument("--z", type=int, default=0, help="Index Z si présent (default: 0)")
    ap.add_argument("--save-tsv", action="store_true", help="Sauver aussi les valeurs par polygone (TSV).")
    ap.add_argument("-th", "--threshold", default="otsu",
                    help="Seuil d’intensité. 'otsu' (défaut) ou une valeur numérique (ex. 1200).")
    return ap.parse_args()


# ------------------------------ Programme ------------------------------- #

def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    print("annot2enrichment: start")
    print(f" - tiff_dir = {args.tiff_dir}")
    print(f" - coco_dir = {args.coco_dir}")
    print(f" - out_dir  = {args.out_dir}")
    print(f" - category = {args.category}")
    print(f" - z        = {args.z}")
    print(f" - threshold= {args.threshold}")

    # Collecte des TIFF
    tiff_paths = sorted(
        chain(Path(args.tiff_dir).glob("*.tif"), Path(args.tiff_dir).glob("*.tiff"))
    )

    all_ratios = []
    rows_for_tsv = []

    cat_tag = sanitize(str(args.category))
    out_tsv = Path(args.out_dir) / f"values_{cat_tag}.tsv"
    with open(out_tsv, "w") as f:
        f.write("Filename\tClass\tX\tY\tMean\n")
        for tif_path in tiff_paths:
            base = tif_path.stem
            print(f"Processing {tif_path.name}")
            anns, id_to_name = load_all_coco_for_base(base, args.category, args.coco_dir)
            if not anns:
                print(f"   ⚠️ No annotations, skipping.")
                continue

            # Get class ID
            selected_class = set()
            if args.category.isdigit():
                selected_class.add(int(args.category))
            else:
                for cid, nm in id_to_name.items():
                    if nm == args.category:
                        selected_class.add(cid)
            if not selected_class:
                print(f"   ⚠️ Unknown category {args.category}.")
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
                if args.signal == "same":
                    signal_ch = ch - 1
                else:
                    signal_ch = int(args.signal) - 1
                    if not (0 <= signal_ch < arr.shape[-3 if arr.ndim == 4 else -3+1]):
                        raise ValueError(f"   ❌ Signal channel {signal_ch} out of bounds for shape {arr.shape}")
                frame = extract_frame(arr, ch=signal_ch, z=args.z)

                # Seuil (Otsu par défaut ou valeur fournie)
                thr_val = None
                if isinstance(args.threshold, str):
                    thr_val = apply_threshold(frame, args.threshold)
                else:
                    thr_val = float(args.threshold)
               
                union_mask = np.zeros((H, W), dtype=bool)
                for i, segs in enumerate(poly_list):
                    mask = polygons_to_mask(segs, (H, W))
                    # Image restreinte au polygone
                    masked_frame = np.zeros_like(frame)
                    masked_frame[mask] = frame[mask]
                    # Sauvegarde de vérification
                    out_png = Path(args.out_dir) / f"{base}_{i}_mask.png"
                    plt.imsave(out_png, masked_frame, cmap="gray")
                    union_mask |= polygons_to_mask(segs, (H, W))
                    seg_mean = measure_polygon_means(frame, [segs], 0)
                    
                    M = mask.sum()
                    if M > 0:
                        ys, xs = np.nonzero(mask)
                        cx, cy = xs.mean(), H - 1 - ys.mean()
                    else:
                        cx, cy = np.nan, np.nan
                    
                    f.write(f"{tif_path.name},{args.category},{cx},{cy},{seg_mean}\n")

    print(f"✅ Data saved in file '{out_tsv}'")


if __name__ == "__main__":
    main()

