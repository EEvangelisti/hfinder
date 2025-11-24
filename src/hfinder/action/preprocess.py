"""
High-level dataset preparation for HFinder.

Overview
--------
- Load per-image class instructions and resolve per-channel operations.
- Threshold or segment channels to produce masks and polygons.
- Convert masks/contours to YOLOv8-compatible polygon labels.
- Compose RGB training images via hue-based fusion of channels.
- Save images/labels to the expected YOLO directory layout.
- Split images into train/val subsets.
- Optionally compute and export MIP (Max Intensity Projection) datasets.

Public API
----------
- prepare_class_inputs(channels, n, c, ratio)
    Build per-frame annotations (polygons) from thresholds or JSON segmentations.
- generate_contours(base, polygons_per_channel, channels, class_ids)
    Render filled/outlined polygons as visual overlays for QA.
- generate_dataset(base, n, c, channels, polygons_per_channel)
    Create RGB composites and YOLO segmentation labels for training.
- split_train_val()
    Move a fraction of training images (and labels) to validation.
- max_intensity_projection_multichannel(img_name, base, stack, polygons_per_channel, class_ids, n, c, ratio)
    Produce a MIP per channel, aggregate polygons across Z, export overlays and dataset.
- generate_training_dataset()
    Orchestrate the full dataset generation flow from input TIFFs.

Notes
-----
- Channel indexing is 1-based throughout (consistent with upstream modules).
- Hue fusion uses deterministic palette rotation when hashing the filename,
  enabling reproducible visuals.
- JSON polygon application to Z-stacks is not implemented (explicit failure).
"""

import os
import cv2
import json
import random
import shutil
import tifffile
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from itertools import chain
from collections import defaultdict, Counter
from hfinder.core import log as HF_log
from hfinder.core import utils as HF_utils
from hfinder.core import palette as HF_palette
from hfinder.core import geometry as HF_geometry
from hfinder.image import processing as HF_ImageOps
from hfinder.image import segmentation as HF_segmentation
from hfinder.session import current as HF_ImageInfo
from hfinder.session import folders as HF_folders
from hfinder.session import settings as HF_settings



def simplify_flat_polygon(flat, epsilon_rel=0.001, min_points=6):
    """
    Simplifie un polygone YOLO (coords normalisées) avec Douglas–Peucker.

    :param flat: [x1, y1, ..., xn, yn] en [0, 1].
    :param epsilon_rel: fraction de la longueur de périmètre utilisée pour eps.
    :param min_points: nombre minimum de points à conserver.
    :return: polygone aplati simplifié, ou original si simplification non pertinente.
    """
    if not flat or len(flat) < 2 * min_points:
        return flat

    pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    cnt = pts.reshape(-1, 1, 2)

    # longueur en coordonnées normalisées (indépendante de la taille en pixels)
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return flat

    eps = epsilon_rel * peri
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

    if approx.shape[0] < min_points:
        # trop simplifié -> on garde la version d'origine
        return flat

    return approx.flatten().tolist()



def resample_flat_polygon(flat, target_points=60):
    """
    Répartit les points d’un polygone à intervalles réguliers sur le périmètre.

    :param flat: [x1, y1, ..., xn, yn] (normalisé).
    :param target_points: nombre de points souhaité (approx.).
    :return: polygone aplati ré-échantillonné.
    """
    if not flat or len(flat) < 6:
        return flat

    pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    n = pts.shape[0]
    if n < 2:
        return flat

    # on ferme explicitement le polygone
    pts_closed = np.vstack([pts, pts[0]])

    # distances cumulées le long du contour
    seg = np.sqrt(((np.diff(pts_closed, axis=0)) ** 2).sum(axis=1))
    d = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = d[-1]
    if perimeter <= 0:
        return flat

    # nb de points utile (évite de gonfler un petit contour)
    K = min(target_points, max(n, 4))
    step = perimeter / K

    new_pts = []
    for k in range(K):
        pos = k * step
        # position sur la chaîne de segments
        idx = np.searchsorted(d, pos) - 1
        if idx < 0:
            idx = 0
        if idx >= len(pts):
            idx = len(pts) - 1

        t_den = d[idx + 1] - d[idx] if (idx + 1 < len(d)) else 1.0
        if t_den <= 1e-8:
            t = 0.0
        else:
            t = (pos - d[idx]) / t_den

        p = (1.0 - t) * pts_closed[idx] + t * pts_closed[idx + 1]
        new_pts.append(p)

    new_pts = np.asarray(new_pts, dtype=np.float32)
    return new_pts.flatten().tolist()


# TODO: set these parameters in HF_settings?
# epsilon_rel : plus grand → contours plus anguleux/moins détaillés.
# target_points : nombre de points cible par polygone.
# min_points : seuil en dessous duquel on ne garde pas un polygone.
def postprocess_polygons(polygons_per_channel,
                         epsilon_rel=0.001,
                         target_points=60,
                         min_points=6):
    """
    Applique simplification + ré-échantillonnage à l’ensemble des polygones.

    :param polygons_per_channel: {ch: [(class_name, [flat_polygons...]), ...]}
    :return: même structure, avec polygones optimisés.
    """
    new_map = defaultdict(list)

    for ch, items in polygons_per_channel.items():
        for class_name, polys in items:
            new_polys = []
            for flat in polys:
                if not flat or len(flat) < 2 * min_points:
                    continue  # on ignore les minuscules artefacts

                flat_simpl = simplify_flat_polygon(flat,
                                                   epsilon_rel=epsilon_rel,
                                                   min_points=min_points)
                flat_resampled = resample_flat_polygon(flat_simpl,
                                                       target_points=target_points)
                new_polys.append(flat_resampled)

            if new_polys:
                new_map[ch].append((class_name, new_polys))

    return new_map



def prepare_class_inputs(channels, n, c, ratio):
    """
    Generate segmentation masks and polygon annotations per class, for each
    frame or image channel, based on class-specific directives.

    For each class (from HF_ImageInfo):
      - If a custom threshold is defined: threshold → mask → contours → polygons.
      - Else if a manual segmentation JSON is provided: load polygons (scaled).
      - Else: apply automatic thresholding → mask → contours → polygons.

    :param channels: Dict mapping 1-based channel/frame index → 2D image array.
    :type channels: dict[int, np.ndarray]
    :param n: Number of Z frames (1 if single plane).
    :type n: int
    :param c: Channels per Z frame.
    :type c: int
    :param ratio: Resize ratio (target_size / original_width) for polygon scaling.
    :type ratio: float
    :return: Map frame index → list of (class_name, [flat_polygons...]).
    :rtype: collections.defaultdict[list]

    :notes:
        - Masks are saved under dataset/masks/, per class/frame.
        - JSON segmentations on Z-stacks (n > 1) fail with an explicit message.
        - Flat polygons are lists like [x1, y1, ..., xn, yn], normalized to [0, 1].
    """
    results = defaultdict(list)
    masks_dir = HF_folders.get_masks_dir()
    base = HF_settings.get("current_image.base")

    for cls in HF_ImageInfo.get_classes():
    
        HF_ImageInfo.set_current_class(cls)
        # Per-class directives
        ch = HF_ImageInfo.get_channel()
        threshold = HF_ImageInfo.get_threshold()
        poly_json = HF_ImageInfo.get_manual_segmentation()

        if threshold is not None:
            # Custom (fixed) thresholding across a frame range
            from_frame = HF_ImageInfo.from_frame(default=0)
            to_frame = HF_ImageInfo.to_frame(default=n)
            for i in range(from_frame // c, to_frame // c + 1):
                frame = i * c + ch
                binary, polygons = HF_segmentation.channel_custom_threshold(channels[frame], threshold)
                results[frame].append((cls, polygons))
                name = f"{base}_{cls}_mask.png" if n == 1 \
                       else f"{base}_frame{frame}_{cls}_mask.png"
                binary_output = os.path.join(masks_dir, name)
                plt.imsave(binary_output, binary, cmap='gray')

        elif poly_json is not None:
            # Load user-provided segmentation polygons (single-plane only)
            if n > 1:
                HF_log.fail(f"File '{base}.tif' - applying user segmentation to Z-stacks has not been implemented yet")
            json_path = os.path.join(HF_settings.get("tiff_dir"), poly_json)
            polygons = HF_segmentation.channel_custom_segment(json_path, ratio)
            results[ch].append((cls, polygons))

        else:
            # Automatic thresholding as a fallback
            from_frame = HF_ImageInfo.from_frame(default=0)
            to_frame = HF_ImageInfo.to_frame(default=n)
            for i in range(from_frame // c, to_frame // c + 1):
                frame = i * c + ch
                binary, polygons = HF_segmentation.channel_auto_threshold(channels[frame])
                results[frame].append((cls, polygons))
                name = f"{base}_{cls}_mask.png" if n == 1 \
                       else f"{base}_frame{frame}_{cls}_mask.png"
                binary_output = os.path.join(masks_dir, name)
                plt.imsave(binary_output, binary, cmap='gray')

    return results



def generate_contours(base, polygons_per_channel, channels, class_ids):
    """
    Draw filled/outlined polygons over grayscale channels and save overlays.

    :param base: Base name used for output files.
    :type base: str
    :param polygons_per_channel: Map channel → [(class_name, [polys...]), ...].
    :type polygons_per_channel: dict[int, list[tuple[str, list[list[float]]]]]
    :param channels: Dict channel → 2D grayscale image.
    :type channels: dict[int, np.ndarray]
    :param class_ids: Class name → integer ID mapping (ignored here except for filtering).
    :type class_ids: dict[str, int]
    :rtype: None
    """
    contours_dir = HF_folders.get_contours_dir()

    for ch_name, polygons in polygons_per_channel.items():
        channel = channels[ch_name]
        h, w = channel.shape
        overlay = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)

        unique_classes = sorted({cn for cn, _ in polygons})
        multi_class = len(unique_classes) > 1
        label_positions = {}

        if multi_class:
            # Stable palette if class set is the same (order-independent)
            hsv_palette = HF_palette.get_random_palette(hash_data="|".join(unique_classes))

            def hsv_to_bgr(h, s, v):
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                return (int(r * 255), int(g * 255), int(b * 255))  # OpenCV uses BGR

            # Map each class to a palette color (cycle if more classes than palette size)
            class_colors = {
                cn: hsv_to_bgr(*hsv_palette[i % len(hsv_palette)])
                for i, cn in enumerate(unique_classes)
            }
        else:
            class_colors = None  # fall back to original behavior


        for class_name, poly in polygons:
            if class_name not in class_ids:
                continue

            for poly in poly:
                # Flat [x1, y1, ..., xn, yn]; ignore degenerate polygons
                if len(poly) < 6:
                    continue  # ignore artifacts (tiny polygons)

                pts = np.array(
                    [(int(poly[i] * w), int(poly[i + 1] * h)) for i in range(0, len(poly), 2)],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                # Choose color (fixed for publication, random for exploration)
                if multi_class:
                    color = class_colors[class_name]
                else:
                    if HF_settings.get("publication"):
                        color = (255, 0, 255)
                    else:
                        color = tuple(random.randint(10, 255) for _ in range(3))

                overlay_copy = overlay.copy()
                # Fill on a copy
                cv2.fillPoly(overlay_copy, [pts], color)
                alpha = 0.3
                overlay = cv2.addWeighted(overlay_copy, alpha, overlay, 1 - alpha, 0)
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)
                if multi_class or not HF_settings.get("publication"):
                    if multi_class:
                        # Stack class names: one line per class, fixed horizontal position
                        if class_name not in label_positions:
                            # Assign next available vertical offset
                            y_offset = 20 + 20 * len(label_positions)
                            label_positions[class_name] = y_offset

                        y_pos = label_positions[class_name]
                        cv2.putText(
                            overlay, class_name, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                        )
                    else:
                        white = (255, 255, 255)
                        cv2.putText(
                            overlay, class_name, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1
                        )

        out_path = os.path.join(contours_dir, f"{base}_{ch_name}_contours.png")
        cv2.imwrite(out_path, overlay)



def generate_dataset(base, n, c, channels, polygons_per_channel):
    """
    Compose RGB training images via hue fusion and export YOLO segmentation labels.

    For each combination of annotated channels (per Z-frame if applicable):
      - Sample optional "noise" channels (never annotated ones; same Z-frame for n>1).
      - Compose a hue-fused RGB image (selected+noise).
      - Save image as JPEG under dataset/images/train/.
      - Export YOLO polygon labels if any annotated polygons exist for the combo.

    :param base: Base filename for images/labels.
    :type base: str
    :param n: Number of Z frames (1 if single plane).
    :type n: int
    :param c: Channels per Z frame.
    :type c: int
    :param channels: Dict channel → 2D grayscale image.
    :type channels: dict[int, np.ndarray]
    :param polygons_per_channel: Dict channel → [(class_name, [polys...]), ...].
    :type polygons_per_channel: dict[int, list[tuple[str, list[list[float]]]]]
    :rtype: None
    """
    img_dir = HF_folders.get_image_train_dir()
    lbl_dir = HF_folders.get_label_train_dir()
    class_ids = HF_settings.load_class_definitions()

    annotated_channels = {ch for ch, polys in polygons_per_channel.items() if polys}
    all_channels = set(channels.keys())

    if HF_settings.get("mode") == "debug":
        print(f"polygons_per_channel.keys() = {polygons_per_channel.keys()}, \
                list(annotated_channels) = {list(annotated_channels)}")

    for ch, img2d in channels.items():
        # 1) Écrire l’image (grayscale → 3 canaux identiques si besoin)
        filename = f"{os.path.splitext(base)[0]}_c{ch:02d}.jpg"
        img_path = os.path.join(img_dir, filename)
        if img2d.ndim == 2:
            img_rgb = np.stack([img2d]*3, axis=-1).astype(np.uint8)
        else:
            img_rgb = img2d  # supposé déjà 3 canaux
        Image.fromarray(img_rgb).save(img_path, "JPEG")

        # 2) Écrire les labels (vides si pas d’annotations)
        annotations = polygons_per_channel.get(ch, [])
        label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
        HF_utils.save_yolo_segmentation_label(label_path, annotations, class_ids)



def split_train_val(validation_frac=0.2, seed=42):

    rng = random.Random(seed)

    img_dir      = HF_folders.get_image_train_dir()
    lbl_dir      = HF_folders.get_label_train_dir()
    img_val_dir  = HF_folders.get_image_val_dir()
    lbl_val_dir  = HF_folders.get_label_val_dir()

    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        HF_log.warn("No training images found to split")
        return

    y_per_img = []          # liste de sets de classes (ex: {0, 2})
    all_classes = set()
    for ip in img_paths:
        name = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, name + ".txt")
        cls_set = set()
        if os.path.isfile(lp):
            with open(lp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tok0 = line.split()[0]
                    try:
                        cls_id = int(tok0)
                        cls_set.add(cls_id)
                    except Exception:
                        continue
        y_per_img.append(cls_set)
        all_classes |= cls_set

    n = len(img_paths)
    target_val = max(1, int(round(validation_frac * n)))

    total_per_class = Counter()
    for s in y_per_img:
        for c in s: total_per_class[c] += 1

    total_all = sum(total_per_class.values())
    class_defs = HF_settings.load_class_definitions(keys='id')
    for c, count in sorted(total_per_class.items()):
        name = class_defs[c]
        pct = 100.0 * count / total_all if total_all else 0
        HF_log.info(f"{name:20s} ({c:2d}) : {count:6d} ({pct:5.2f}%)")

    target_per_class_val = {c: validation_frac * total_per_class[c] for c in total_per_class}

    # 3) Structures pour l'itérative stratification
    #    - pour chaque classe → indices d’images la contenant
    indices_per_class = {c: [] for c in all_classes}
    for i, s in enumerate(y_per_img):
        for c in s:
            indices_per_class[c].append(i)
    for c in indices_per_class:
        rng.shuffle(indices_per_class[c])

    assigned = [False] * n
    in_val   = [False] * n
    cur_per_class_val = Counter()
    val_count = 0

    # Fonction utilitaire: besoin restant par classe dans val
    def remaining_need(c):
        return target_per_class_val.get(c, 0.0) - cur_per_class_val[c]

    # 4) Étape A — satisfaire au mieux les classes rares d’abord
    #    Boucle tant qu’on peut améliorer et qu’il reste du budget val
    #    Heuristique: prendre la classe avec plus grand "need" restant,
    #    puis l’image la plus "rare" (moins d’étiquettes).
    while val_count < target_val:
        # Classe la plus "sous-représentée" côté val
        candidates_classes = [c for c in all_classes if remaining_need(c) > 1e-9]
        if not candidates_classes:
            break
        c_star = max(candidates_classes, key=lambda c: remaining_need(c))

        # Parmi ses images non assignées, prendre celle avec le moins de labels (rare)
        pool = [i for i in indices_per_class[c_star] if not assigned[i]]
        if not pool:
            # rien pour cette classe → on marquera comme impossible plus bas
            # On la “sature” pour éviter boucle infinie
            all_classes.remove(c_star)
            continue

        i_star = min(pool, key=lambda i: len(y_per_img[i]) if y_per_img[i] else 0)

        # Assigner à val
        assigned[i_star] = True
        in_val[i_star] = True
        val_count += 1
        for c in y_per_img[i_star]:
            cur_per_class_val[c] += 1

    # 5) Étape B — remplir le reste du budget val de façon douce (distance aux cibles)
    def gain_if_val(i):
        # somme des besoins pour les classes présentes dans i
        return sum(max(0.0, remaining_need(c)) for c in y_per_img[i])

    remaining = [i for i in range(n) if not assigned[i]]
    # Donner la priorité aux images qui améliorent le plus la couverture des classes
    remaining.sort(key=lambda i: (gain_if_val(i), -len(y_per_img[i])), reverse=True)

    for i in remaining:
        if val_count >= target_val:
            break
        # si l’image est négative, autoriser mais donner moins de priorité (déjà géré par tri)
        assigned[i] = True
        in_val[i] = True
        val_count += 1
        for c in y_per_img[i]:
            cur_per_class_val[c] += 1

    # 6) Le reste va en train
    for i in range(n):
        if not assigned[i]:
            in_val[i] = False

    # 7) Déplacement des fichiers
    moved_val, moved_train = 0, 0
    for i, ip in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, name + ".txt")
        dest_img_dir = img_val_dir if in_val[i] else img_dir  # rester sur place pour train
        dest_lbl_dir = lbl_val_dir if in_val[i] else lbl_dir

        if in_val[i]:
            # déplacer vers val
            shutil.move(ip, os.path.join(dest_img_dir, os.path.basename(ip)))
            if os.path.exists(lp):
                shutil.move(lp, os.path.join(dest_lbl_dir, os.path.basename(lp)))
            else:
                # créer un .txt vide si manquant, pour cohérence
                open(os.path.join(dest_lbl_dir, name + ".txt"), "a").close()
            moved_val += 1
        else:
            # s'assurer que le label existe (éventuellement vide)
            if not os.path.exists(lp):
                open(lp, "a").close()
            moved_train += 1

    # 8) Petit bilan
    def count_dir(lbl_path):
        cnt = Counter()
        for p in glob(os.path.join(lbl_path, "*.txt")):
            with open(p) as f:
                for line in f:
                    line=line.strip()
                    if line:
                        try: cnt[int(line.split()[0])] += 1
                        except: pass
        return cnt

    val_counts   = count_dir(lbl_val_dir)
    train_counts = count_dir(lbl_dir)

    HF_log.info(f"Split done. val images={moved_val}, train images={moved_train}")
    HF_log.info(f"Per-class (train): {dict(train_counts)}")
    HF_log.info(f"Per-class (val)  : {dict(val_counts)}")



def max_intensity_projection_multichannel(img_name, base, stack, polygons_per_channel, class_ids, n, c, ratio):
    """
    Build a Max Intensity Projection (MIP) across Z and export a mini-dataset.

    Steps per channel index 0..c-1:
      - Compute the MIP across n frames → resize to target size.
      - Aggregate polygons across all Z-slices for each class.
      - Fill a global mask per class and convert to YOLO polygons.
      - Save masks and overlays; export a small MIP-based dataset.

    :param img_name: Original image filename (for bookkeeping).
    :type img_name: str
    :param base: Base name for outputs (suffix "_MIP" is added).
    :type base: str
    :param stack: Original image stack (ndarray with Z).
    :type stack: np.ndarray
    :param polygons_per_channel: Dict channel_idx → [(class_name, [polys...]), ...].
    :type polygons_per_channel: dict[int, list[tuple[str, list[list[float]]]]]
    :param class_ids: Mapping class name → ID.
    :type class_ids: dict[str, int]
    :param n: Number of Z frames.
    :type n: int
    :param c: Channels per frame.
    :type c: int
    :param ratio: Resize factor used elsewhere (kept for parity).
    :type ratio: float
    :rtype: None

    :notes:
        - This routine assumes square targets (size × size).
        - It aggregates instance polygons across Z before re-vectorizing.
    """
    mip = np.max(stack, axis=0)   # shape: (C, H, W)
    stacked_channels = [
        cv2.resize(mip[ch], (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        for ch in range(c)
    ]
    size = HF_settings.get("size")
    assert (stacked_channels[0].shape == (size, size))

    masks_dir = HF_folders.get_masks_dir()
    contours_dir = HF_folders.get_contours_dir()

    # For each channel of the MIP, merge polygons from all Z-slices
    polygons_per_stacked_channel = defaultdict(list)
    for ch in range(c):
        # 1-based indices of frames for this channel across Z: 1, 1+c, 1+2c, ...
        indices = [j + 1 for j in range(ch, n * c, c)]  # 1, 1+c, 1+2c, ...
        
        # Collect per-slice polygons for the channel: [(label, [polys...]), ...]
        polygons_subset = [polygons_per_channel.get(idx, []) for idx in indices]

        # Build a fused mask per class from all Z-slice polygons
        allowed_items = [(x, y) for x, y in class_ids.items() if HF_ImageInfo.allows_MIP_generation(x)]
        for class_name, class_id in allowed_items:
            # Accumulate polygons (pixel coords) for this class across slices
            all_polys_px = []
            for per_slice in polygons_subset:
                for label, polys_list in per_slice:
                    if label != class_name:
                        continue
                    # polys_list est une liste de polygones plats
                    for flat in polys_list:
                        if not flat: 
                            continue
                        # Convert normalized flat polygon to pixel coordinates
                        pts = HF_geometry.flat_to_pts_xy(flat)   # (N, 2)
                        if pts.shape[0] >= 3:
                            all_polys_px.append(pts)

            # If nothing accumulated, skip this class/channel
            if not all_polys_px:
                continue

            # Fused binary mask for the class on the MIP channel
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(mask, all_polys_px, 255)

            clean_mask = HF_segmentation.noise_and_gaps(mask)
            # Persist the fused class mask (for QA or reuse)
            mask_path = os.path.join(masks_dir, f"{base}_MIP_{class_name}_mask.png")
            cv2.imwrite(mask_path, clean_mask)
            # Extract refined contours and convert to YOLO polygons
            final_contours = HF_segmentation.mask_to_polygons(clean_mask)
            yolo_polygons = HF_geometry.contours_to_yolo_polygons(final_contours)
 
            ch_key = ch + 1   # 1-based
            polygons_per_stacked_channel[ch_key].append((class_name, yolo_polygons))

    # Compose overlays and export a small MIP dataset
    stacked_channels_dict = {i+1: stacked_channels[i] for i in range(c)}
    generate_contours(
        base + "_MIP",
        polygons_per_stacked_channel, 
        stacked_channels_dict,
        class_ids
    )  

    generate_dataset(
        base + "_MIP",
        n=1, c=c,
        channels=stacked_channels_dict,
        polygons_per_channel=polygons_per_stacked_channel
    )



def generate_training_dataset():
    """
    End-to-end dataset generation from a folder of TIFFs.

    Steps:
      - Discover input images under `tiff_dir`.
      - Initialize class instructions and write dataset YAML.
      - For each image:
          * Validate shape; resize/stack channels.
          * Create class-specific masks/annotations.
          * Save overlays and fused RGB training examples (+ labels).
          * Optionally export MIP-based dataset if multiple Z-frames.
      - Finally split train/val per `validation_frac`.

    :rtype: None
    """
    data_dir = HF_settings.get("tiff_dir")
    image_paths = sorted(glob(os.path.join(data_dir, "*.tif")))
    
    class_ids = HF_settings.load_class_definitions()
    HF_folders.write_yolo_yaml(class_ids)
    HF_ImageInfo.initialize()

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        HF_ImageInfo.set_current_image(img_name)
        img_base = HF_ImageInfo.get_current_base()

        if not HF_ImageInfo.image_has_instructions():
            HF_log.warn(f"Skipping file {img_name} - no annotations")
            continue

        img = tifffile.imread(img_path)
        if not HF_geometry.is_valid_image_format(img):
            HF_log.warn(f"Skipping file {img_name}, wrong shape {img.shape}")
            continue

        # Resize channels to the configured size; get ratio and (n, c)
        channels, ratio, (n, c) = HF_ImageOps.resize_multichannel_image(img)   
        polygons_per_channel = prepare_class_inputs(channels, n, c, ratio)
        
        # Post-traitement des polygones pour simplifier et homogénéiser les contours
        polygons_per_channel = postprocess_polygons(
            polygons_per_channel
            #epsilon_rel=HF_settings.get("poly_epsilon_rel", 0.001),
            #target_points=HF_settings.get("poly_target_points", 60),
            #min_points=HF_settings.get("poly_min_points", 6),
        )
        
        # QA overlays then dataset generation
        generate_contours(img_base, polygons_per_channel, channels, class_ids)     
        generate_dataset(img_base, n, c, channels, polygons_per_channel)

        # Optional MIP export when multiple Z-slices exist
        # FIXME: currently deactivated because it is not satisfactory.
        if n > 1 and False:
            max_intensity_projection_multichannel(
                img_name, img_base, img, polygons_per_channel,
                class_ids, n, c, ratio
            )

    # Split train/val at the end so both single-plane and MIP outputs are included
    split_train_val()
