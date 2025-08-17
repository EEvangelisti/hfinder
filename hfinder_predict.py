"""
Prediction from TIFFs with channel-fusion ensembling and vote consolidation.

This module implements the prediction stage of HFinder:
- TIFF files are converted into fused RGB images using the same 
  channel-combination logic as during training.
- YOLOv8 predictions are run across all fused images.
- Detections are consolidated in two stages:
    1. Cross-class filtering (objects cannot belong to two classes).
    2. Intra-class consolidation (IoU-based voting).
- Results are exported in two formats:
    - consolidated.json (custom format with votes, boxes, polygons),
    - coco.json (COCO-compatible annotations with boxes + polygons).

Public API
----------
- run(): Perform predictions from TIFFs, consolidate detections, and save results.
- resolve_device(raw): Map user-specified device string/int to a valid PyTorch device.
- build_fusions_for_tiff(): Generate fused RGB images from a multichannel TIFF.
- consolidate_boxes_two_stage(): Apply cross-class filtering and intra-class consolidation.
- coco_skeleton(): Build an empty COCO structure with given categories.
"""


import os
import cv2
import json
import math
import torch
import random
import tifffile
import numpy as np
from PIL import Image
from glob import glob
from collections import defaultdict, Counter
from ultralytics import YOLO

import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings
import hfinder_imageops as HFinder_ImageOps
import hfinder_palette as HFinder_palette
import hfinder_utils as HFinder_utils



def resolve_device(raw):
    """
    Resolve a device string or integer to a PyTorch-compatible device.

    :param raw: Raw device specifier ("cpu", "auto", int, etc.).
    :type raw: str | int | None
    :return: "cpu" or CUDA device index.
    :rtype: str | int
    """
    if isinstance(raw, str) and raw.strip().lower() == "cpu":
        return "cpu"
    if isinstance(raw, str) and raw.strip().lower() in ("auto", ""):
        raw = None
    if raw is None:
        return 0 if torch.cuda.is_available() else "cpu"
    try:
        idx = int(raw)
        return idx if torch.cuda.is_available() and (0 <= idx < torch.cuda.device_count()) else "cpu"
    except (TypeError, ValueError):
        return "cpu"



def build_whitelist_ids(whitelist_pairs_names, name_to_id):
    """
    Convert class-name pairs into a set of class-ID pairs (order-agnostic).

    Each input pair (e.g., ["haustoria", "hyphae"]) is mapped to their 
    corresponding class IDs using `name_to_id`. We wrap the two IDs into a 
    `frozenset`, which makes the pair:
      - immutable (safe as a set element or dict key),
      - order-agnostic (frozenset({a,b}) == frozenset({b,a})).

    This way, (A,B) and (B,A) are treated as the same allowed overlay pair.

    :param whitelist_pairs_names: List of class name pairs, e.g.
                                  [["haustoria","hyphae"], ["nuclei","chloroplasts"]].
    :type whitelist_pairs_names: list[list[str]]
    :param name_to_id: Mapping from class name to class ID, e.g. {"nuclei": 0, "chloroplasts": 1}.
    :type name_to_id: dict[str, int]
    :return: Set of frozensets, each containing two class IDs that are whitelisted.
    :rtype: set[frozenset[int]]
    """
    wl = set()
    for a, b in whitelist_pairs_names:
        if a in name_to_id and b in name_to_id:
            wl.add(frozenset({name_to_id[a], name_to_id[b]}))
    return wl



def area_xyxy(b):
    """
    Compute the area (in pixels²) of a bounding box [x1, y1, x2, y2].

    The result is max(0, x2−x1) * max(0, y2−y1), ensuring a non-negative area
    even if coordinates are partially inverted.

    :param b: Bounding box in xyxy format.
    :type b: list[float] | np.ndarray
    :return: Non-negative box area in pixel units.
    :rtype: float
    """
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)



def iou_xyxy(a, b):
    """
    Compute IoU between two bounding boxes in xyxy format.

    :param a: [x1,y1,x2,y2].
    :type a: list[float] | np.ndarray
    :param b: [x1,y1,x2,y2].
    :type b: list[float] | np.ndarray
    :return: Intersection-over-Union.
    :rtype: float
    """
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / (union + 1e-9)



def cross_class_filter(all_dets, iou_inter, policy, whitelist_ids):
    """
    Apply cross-class suppression to raw detections.

    :param all_dets: List of detections ({cls, conf, xyxy, segs}).
    :type all_dets: list[dict]
    :param iou_inter: IoU threshold for inter-class conflict.
    :type iou_inter: float
    :param policy: Strategy ("keep_best", "drop_both", "allow_if_whitelisted").
    :type policy: str
    :param whitelist_ids: Allowed class-ID pairs.
    :type whitelist_ids: set[frozenset[int]]
    :return: Filtered list of detections.
    :rtype: list[dict]
    """
    if not all_dets:
        return []

    # order by confidence desc, then has polygons, then area (tie-breakers)
    order = sorted(
        range(len(all_dets)),
        key=lambda i: (all_dets[i]["conf"], bool(all_dets[i].get("segs")), area_xyxy(all_dets[i]["xyxy"])),
        reverse=True
    )

    kept = []
    suppressed = [False] * len(all_dets)

    def pair_allowed(ci, cj):
        return frozenset({ci, cj}) in whitelist_ids

    for i in order:
        if suppressed[i]:
            continue
        kept.append(i)
        bi = np.asarray(all_dets[i]["xyxy"], dtype=np.float32)
        ci = int(all_dets[i]["cls"])
        for j in order:
            if j <= i or suppressed[j]:
                continue
            cj = int(all_dets[j]["cls"])
            if cj == ci:
                continue  # same class handled later (intra-class)
            bj = np.asarray(all_dets[j]["xyxy"], dtype=np.float32)
            if iou_xyxy(bi, bj) > iou_inter:
                # if whitelist says allowed, keep both
                if policy == "allow_if_whitelisted" and pair_allowed(ci, cj):
                    continue
                # otherwise enforce chosen strategy
                if policy == "drop_both":
                    suppressed[j] = True
                    suppressed[i] = True
                    break  # current i is dropped; stop suppressing with it
                else:  # keep_best (i precedes j by confidence)
                    suppressed[j] = True

    survivors = [all_dets[i] for i in range(len(all_dets)) if not suppressed[i]]
    return survivors



def intra_class_consolidate_keep_best(dets, iou_intra=0.5, min_votes=2):
    """
    Consolidate detections within each class by IoU voting.

    :param dets: Detections (same class handled separately).
    :type dets: list[dict]
    :param iou_intra: IoU threshold for grouping same-class detections.
    :type iou_intra: float
    :param min_votes: Minimum votes to keep a cluster.
    :type min_votes: int
    :return: Consolidated detections per class.
    :rtype: list[dict]
    """

    by_class = defaultdict(list)
    for d in dets:
        by_class[int(d["cls"])].append(d)

    out = []
    for cls, items in by_class.items():
        clusters = []
        for det in items:
            matched = False
            for cl in clusters:
                if iou_xyxy(det["xyxy"], cl["center"]) >= iou_intra:
                    cl["members"].append(det)
                    cl["votes"] += 1
                    cl["center"] = (cl["center"] * (cl["votes"] - 1) + det["xyxy"]) / cl["votes"]
                    matched = True
                    break
            if not matched:
                clusters.append({"center": det["xyxy"].astype(np.float32).copy(),
                                 "members": [det], "votes": 1})

        for cl in clusters:
            if cl["votes"] < min_votes:
                continue
            best = max(
                cl["members"],
                key=lambda d: (float(d["conf"]), bool(d.get("segs")), area_xyxy(d["xyxy"]))
            )
            out.append({
                "cls": cls,
                "votes": int(cl["votes"]),
                "conf": float(best["conf"]),
                "conf_mean": float(best["conf"]),  # <-- add this for compatibility
                "xyxy": best["xyxy"].tolist(),
                "segs": best.get("segs", [])
            })
    return out
    
    

def consolidate_boxes_two_stage(all_dets, iou_inter=0.5, iou_intra=0.5,
                                min_votes=2, whitelist_ids=frozenset(),
                                policy="keep_best"):
    """
    Two-stage consolidation of detections:
      A) Cross-class filtering,
      B) Intra-class IoU-based consolidation.

    :param all_dets: Raw detections ({cls, conf, xyxy, segs}).
    :type all_dets: list[dict]
    :param iou_inter: IoU threshold for cross-class conflicts.
    :type iou_inter: float
    :param iou_intra: IoU threshold for intra-class consolidation.
    :type iou_intra: float
    :param min_votes: Minimum votes for a cluster to be kept.
    :type min_votes: int
    :param whitelist_ids: Set of class-ID pairs allowed to overlap.
    :type whitelist_ids: set[frozenset[int]]
    :param policy: Cross-class conflict policy.
    :type policy: str
    :return: Consolidated detections.
    :rtype: list[dict]
    """

    survivors = cross_class_filter(
        all_dets,
        iou_inter=float(iou_inter),
        policy=(policy or "keep_best").lower(),
        whitelist_ids=whitelist_ids
    )

    # 2) Intra-class consolidation
    consolidated = intra_class_consolidate_keep_best(
        survivors,
        iou_intra=float(iou_intra),
        min_votes=int(min_votes)
    )
    return consolidated



def build_fusions_for_tiff(tif_path, out_dir, rng=None):
    """
    Build fused RGB images from a multichannel TIFF.

    :param tif_path: Path to input TIFF.
    :type tif_path: str
    :param out_dir: Directory where fusions are saved.
    :type out_dir: str
    :param rng: Optional random generator for reproducibility.
    :type rng: random.Random | None
    :return: (list of (img_path, channels), (H,W), (n,c))
    :rtype: tuple[list[tuple[str,tuple[int]]], tuple[int,int], tuple[int,int]]
    """
    if rng is None:
        rng = random.Random(0)

    img = tifffile.imread(tif_path)
    channels_dict, ratio, (n, c) = HFinder_ImageOps.resize_multichannel_image(img)  # {1..n*c: (H,W)}
    all_channels = sorted(channels_dict.keys())

    os.makedirs(out_dir, exist_ok=True)

    fusions = []
    base = os.path.splitext(os.path.basename(tif_path))[0]

    # Combinaisons 1-3 par tranche Z, comme pendant l'entraînement
    combos = HFinder_utils.power_set(all_channels, n, c)  # déjà par tranche si n>1

    for combo in combos:
        combo = tuple(sorted(combo))
        # Canaux "bruit" candidats = canaux de la même tranche non dans combo
        noise_candidates = list(set(all_channels) - set(combo))
        if n > 1:
            ref_ch = min(combo)
            series_index = (ref_ch - 1) // c
            allowed_noise = [series_index * c + i + 1 for i in range(c)]
            noise_candidates = sorted(list(set(noise_candidates) & set(allowed_noise)))
        # Échantillonnage aléatoire 0..K bruits
        k = rng.randint(0, len(noise_candidates)) if noise_candidates else 0
        noise = rng.sample(noise_candidates, k) if k > 0 else []

        # Palette déterministe (hash sur le nom de fichier de sortie)
        fname = f"{base}_" + "_".join(map(str, combo)) + ".jpg"
        palette = HFinder_palette.get_random_palette(hash_data=fname)

        rgb = HFinder_ImageOps.compose_hue_fusion(
            channels=channels_dict,
            selected_channels=list(combo),
            palette=palette,
            noise_channels=noise
        )

        out_path = os.path.join(out_dir, fname)
        Image.fromarray(rgb).save(out_path, "JPEG")
        fusions.append((out_path, combo))

    # Dimensions (H,W)
    H, W = next(iter(channels_dict.values())).shape
    return ratio, fusions, (H, W), (n, c)



def coco_skeleton(categories):
    """
    Create an empty COCO JSON structure.

    :param categories: Mapping name->id.
    :type categories: dict[str,int]
    :return: COCO dict with categories.
    :rtype: dict
    """
    cats = [{"id": cid, "name": name, "supercategory": "hf"} 
            for name, cid in sorted(categories.items(), key=lambda x: x[1])]
    return {"images": [], "annotations": [], "categories": cats}



def poly_area_xy(poly):
    """
    Compute the polygon area (in pixels²) from a flattened [x1, y1, x2, y2, ...] list.
    Uses the shoelace formula. Degenerate inputs (fewer than 3 points) return 0.0.

    :param poly: Flattened polygon coordinates [x1, y1, x2, y2, ...].
    :type poly: list[float] | np.ndarray
    :return: Non-negative polygon area in pixel units.
    :rtype: float
    """
    if not poly or len(poly) < 6: 
        return 0.0
    it = iter(poly)
    pts = [(float(x), float(next(it))) for x in it]
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return 0.5 * abs(sum(x[i] * y[(i + 1) % len(pts)] - 
           x[(i + 1) % len(pts)] * y[i] for i in range(len(pts))))



def bbox_xyxy_to_xywh(b):
    """
    Compute the area (in pixels²) of a bounding box [x1, y1, x2, y2].

    The result is max(0, x2−x1) * max(0, y2−y1), ensuring a non-negative area
    even if coordinates are partially inverted.

    :param b: Bounding box in xyxy format.
    :type b: list[float] | np.ndarray
    :return: Non-negative box area in pixel units.
    :rtype: float
    """
    x1, y1, x2, y2 = map(float, b)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]



def polys_from_xy(inst_xy):
    """
    Convert YOLO mask polygon(s) into COCO-style flattened lists.

    A YOLO result may store instance polygons either as:
      - a single (N,2) numpy array of x,y coordinates, or
      - a list of such arrays (for multiple disjoint polygons).

    This function normalizes the input into a list of flattened
    [x1, y1, x2, y2, ..., xn, yn] lists.

    :param inst_xy: A (N,2) array or list of arrays with polygon vertices.
    :type inst_xy: numpy.ndarray | list[numpy.ndarray] | None
    :return: A list of polygons, each represented as a flattened list of floats.
    :rtype: list[list[float]]
    """
    if inst_xy is None:
        return []
    polys = inst_xy if isinstance(inst_xy, (list, tuple)) else [inst_xy]
    flats = []
    for poly in polys:
        arr = np.asarray(poly)
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue
        flats.append(arr.reshape(-1).tolist())
    return flats



def polys_from_mask_i(result_obj, i):
    """
    Extract polygon(s) from the i-th mask of a YOLO result object.

    If YOLO does not provide polygons (or if they are empty),
    this function derives them from the binary mask using OpenCV contours.
    The mask is first morphologically closed to smooth small gaps.

    :param result_obj: YOLO result object with .masks.data attribute.
    :type result_obj: ultralytics.engine.results.Results
    :param i: Index of the instance mask to convert.
    :type i: int
    :return: A list of polygons (flattened [x1,y1,...]) extracted from the mask.
    :rtype: list[list[float]]
    """
    if getattr(result_obj, "masks", None) is None or getattr(result_obj.masks, "data", None) is None:
        return []
    m_i = result_obj.masks.data[i].detach().cpu().numpy().astype(np.uint8) * 255
    m_i = cv2.morphologyEx(m_i, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flats = []
    for c in cnts:
        if c.shape[0] < 3 or cv2.contourArea(c) < 10:
            continue
        flats.append(c.reshape(-1, 2).astype(float).reshape(-1).tolist())
    return flats



def rescale_box_xyxy(box, scale_factor):
    """
    Rescale a bounding box from resized space back to original space.

    This function takes a bounding box in [x1, y1, x2, y2] format (resized space)
    and multiplies all values by the scale factor. The result is a bounding box
    expressed in the original TIFF coordinate system.

    :param box: Bounding box coordinates [x1, y1, x2, y2].
    :type box: list[float] | tuple[float, float, float, float]
    :param scale_factor: Factor to convert from resized to original coordinates
                         (usually 1.0 / resize_ratio).
    :type scale_factor: float
    :return: Rescaled bounding box in original coordinates.
    :rtype: list[float]
    """
    return [float(v) * scale_factor for v in box]



def rescale_seg_flat(seg, scale_factor):
    """
    Rescale a flattened polygon segmentation from resized space back to original space.

    This function takes a segmentation polygon represented as a flat list
    of alternating x and y coordinates, e.g. [x1, y1, x2, y2, ...], and multiplies
    all values by the scale factor. The result is a polygon expressed in the
    original TIFF coordinate system.

    :param seg: Flattened polygon segmentation (list of floats).
    :type seg: list[float]
    :param scale_factor: Factor to convert from resized to original coordinates
                         (usually 1.0 / resize_ratio).
    :type scale_factor: float
    :return: Rescaled polygon segmentation in original coordinates.
    :rtype: list[float]
    """
    return [float(v) * scale_factor for v in seg]



def run():
    """
    Run full prediction pipeline:
      - Load YOLO model.
      - Generate fused images from TIFFs.
      - Predict on fusions.
      - Consolidate detections.
      - Save outputs (consolidated.json, coco.json).

    :rtype: None
    """
    # Weights / model
    weights = HFinder_settings.get("weights")
    if not weights:
        HFinder_log.fail("Weights needed to perform predictions")
    if not os.path.exists(weights):
        HFinder_log.fail(f"Weights file not found: {weights}")
    model = YOLO(weights)

    # Settings
    conf = HFinder_settings.get("conf") or 0.25
    imgsz = HFinder_settings.get("size") or 640
    device = resolve_device(HFinder_settings.get("device"))
    batch = HFinder_settings.get("batch") or 8
    iou_vote = float(HFinder_settings.get("vote_iou") or 0.5)
    min_votes = int(HFinder_settings.get("vote_min") or 2)

    # Retrieve class names from a YAML file (e.g., generated during training)
    yaml_path = HFinder_settings.get("yaml")
    if not os.path.isfile(yaml_path):
        HFinder_log.fail(f"YAML file not found: {yaml_path}")
    class_ids = HFinder_utils.load_class_definitions_from_yaml(yaml_path)  # {"name": id}

    # Cross-class and inter-class settings
    cross_iou = float(HFinder_settings.get("cross_iou")) or 0.5
    overlay_policy = (HFinder_settings.get("overlay_policy") or "keep_best").lower()
    try:
        wl_raw = HFinder_settings.get("overlay_whitelist") or "[]"
        wl_pairs = json.loads(wl_raw)  # e.g. [["haustoria","hyphae"]]
    except Exception:
        wl_pairs = []
    whitelist_ids = build_whitelist_ids(wl_pairs, class_ids)  # set of frozenset({idA,idB})
    if wl_pairs and not whitelist_ids:
        HFinder_log.warn("overlay_whitelist provided but no names matched class IDs from YAML.")
    HFinder_log.info(f"Overlay policy={overlay_policy}, IoU={cross_iou}, whitelist_pairs={len(wl_pairs)}")

    # Torch settings
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        torch.set_num_interop_threads(1)

    # Input TIFFs
    input_folder = HFinder_settings.get("tiff_dir")
    if not input_folder or not os.path.isdir(input_folder):
        HFinder_log.fail(f"Invalid 'tiff_dir': {input_folder}")
    tiffs = sorted(glob(os.path.join(input_folder, "*.tif")) + glob(os.path.join(input_folder, "*.tiff")))
    if not tiffs:
        HFinder_log.warn(f"No TIFF files found in {input_folder}")
        return

    project = HFinder_folders.get_runs_dir()
    HFinder_log.info(f"Predicting on {len(tiffs)} TIFF(s) | conf={conf} | imgsz={imgsz} | device={device}")

    # ---- TIFF loop -----------------------------------------------------------
    for tif_path in tiffs:
        tif_file = os.path.basename(tif_path)
        tif_base = os.path.splitext(tif_file)[0]
        out_dir = os.path.join(project, "predict", tif_base)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Generate fused RGB images (same logic as training)
        rng = random.Random(tif_base)  # deterministic per TIFF
        ratio, fusions, (H, W), (n, c) = build_fusions_for_tiff(tif_path, out_dir, rng=rng)
        orig_W = int(round(W / ratio))
        orig_H = int(round(H / ratio))
        scale_factor = 1.0 / ratio
        fusion_paths = [p for p, _ in fusions]
        if not fusion_paths:
            HFinder_log.warn(f"[{tif_base}] No fused images generated; skipping")
            continue
        HFinder_log.info(f"[{tif_base}] Generated {len(fusion_paths)} fused images (n={n}, c={c})")

        # 2) Run predictions on all fusions
        results = model.predict(
            source=fusion_paths,
            batch=batch,
            save=True,            # saves overlays into runs/predict/<tif_base>/
            project=project,
            name=os.path.join("predict", tif_base),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
            retina_masks=True     # better polygon quality
        )

        # 3) Collect detections (boxes + polygons) for IoU voting
        all_dets = []
        for img_path, r in zip(fusion_paths, results):
            if getattr(r, "boxes", None) is None or r.boxes is None or r.boxes.shape[0] == 0:
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)

            xy_polys = None
            if getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None:
                xy_polys = r.masks.xy  # can be list-of-arrays or single (N,2) per instance

            for i, (bb, sc, cc) in enumerate(zip(xyxy, confs, clss)):
                # Clamp to (W,H)
                x1 = float(max(0.0, min(bb[0], W)))
                y1 = float(max(0.0, min(bb[1], H)))
                x2 = float(max(0.0, min(bb[2], W)))
                y2 = float(max(0.0, min(bb[3], H)))
                det = {
                    "cls": int(cc),
                    "conf": float(sc),
                    "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "segs": []
                }

                # 1) Try masks.xy (handles (N,2) OR list of (N,2))
                if xy_polys is not None and i < len(xy_polys) and xy_polys[i] is not None:
                    det["segs"] = polys_from_xy(xy_polys[i])

                # 2) Fallback: contours from binary mask
                if not det["segs"]:
                    det["segs"] = polys_from_mask_i(r, i)

                all_dets.append(det)

        # 4) Consolidation (IoU voting) — consolidate_boxes keeps best_with_segs when available
        consolidated = consolidate_boxes_two_stage(
            all_dets,
            iou_inter=cross_iou,
            iou_intra=iou_vote,
            min_votes=min_votes,
            whitelist_ids=whitelist_ids,
            policy=overlay_policy
        )

        # 5) Save JSONs
        # 5a) consolidated.json — in ORIGINAL coords
        for det in consolidated:
            det["xyxy"] = rescale_box_xyxy(det["xyxy"], scale_factor)
            if det.get("segs"):
                det["segs"] = [rescale_seg_flat(seg, scale_factor) for seg in det["segs"]]
        summary = {
            "tiff": tif_file,
            "img_size": [int(W * scale_factor), int(H * scale_factor)],
            "vote_iou": iou_vote,
            "vote_min": min_votes,
            "detections": consolidated  # [{cls, votes, conf_mean, xyxy, segs}]
        }
        
        with open(os.path.join(input_folder, f"{tif_base}.consolidated.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # 5b) coco.json — also ORIGINAL coords
        # coco skeleton: {"images": [], "annotations": [], "categories":[...]}
        coco = coco_skeleton(class_ids)
        image_id = 1
        ref_img_rel = os.path.basename(fusion_paths[0])
        coco["images"].append({
            "id": image_id, 
            "file_name": tif_file,
            "width": int(W * scale_factor),
            "height": int(H * scale_factor)
        })

        ann_id = 1
        for det in consolidated:
            bbox_xywh = bbox_xyxy_to_xywh(det["xyxy"])
            if det.get("segs"):
                area = float(sum(poly_area_xy(seg) for seg in det["segs"]))
                segmentation = det["segs"]
            else:
                area = float(bbox_xywh[2] * bbox_xywh[3])
                segmentation = []

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(det["cls"]),
                "bbox": [round(v, 2) for v in bbox_xywh],
                "area": round(area, 2),
                "segmentation": segmentation,
                "iscrowd": 0,
                "confidence": round(det["conf_mean"], 4),
                "votes": int(det["votes"])
            })
            ann_id += 1

        with open(os.path.join(input_folder, f"{tif_base}.coco.json"), "w") as f:
            json.dump(coco, f, indent=2)

        # 6) Logs
        counts = Counter([d["cls"] for d in consolidated])
        HFinder_log.info(
            f"[{tif_base}] Consolidated {sum(counts.values())} detections across "
            f"{len(set(counts.elements()))} classes; COCO & consolidated saved."
        )

