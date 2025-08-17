# Prediction from TIFFs with channel-fusion ensembling + vote consolidation

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


# -----------------------
# Device resolution
# -----------------------
def resolve_device(raw):
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


# -----------------------
# IoU + consolidation
# -----------------------
def build_whitelist_ids(whitelist_pairs_names, name_to_id):
    """
    Convert [["haustoria","hyphae"], ...] to set of frozenset({idA,idB}).
    Order-agnostic.
    """
    wl = set()
    for a, b in whitelist_pairs_names:
        if a in name_to_id and b in name_to_id:
            wl.add(frozenset({name_to_id[a], name_to_id[b]}))
    return wl


def iou_xyxy(a, b):
    """IoU entre deux boîtes xyxy (np.array [x1,y1,x2,y2])."""
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b 
    return inter / (union - inter + 1e-9)


def area_xyxy(b):
    x1,y1,x2,y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def cross_class_filter(all_dets, iou_inter, policy, whitelist_ids):
    """
    all_dets: list of raw detections (each: {"cls": int, "conf": float, "xyxy": np.ndarray, "segs": [...]})
    Return a filtered list after applying cross-class policy.

    policy:
      - "keep_best" : keep higher-confidence among conflicting pair.
      - "drop_both" : drop both in a conflicting pair.
      - "allow_if_whitelisted" : if pair allowed, keep both; else apply "keep_best".
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
    def iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter + 1e-9
        return inter / denom

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
    
    

def consolidate_boxes_two_stage(
    all_dets,
    iou_inter=0.5,      # cross-class IoU
    iou_intra=0.5,      # intra-class IoU
    min_votes=2,
    whitelist_ids=frozenset(),
    policy="keep_best"
):
    """
    Two-stage consolidation:
      A) Cross-class suppression (objects cannot be two different classes):
         if IoU > iou_inter between different classes, keep the higher-confidence one.
      B) Intra-class consolidation:
         cluster by IoU >= iou_intra; for each cluster (votes >= min_votes),
         return the highest-confidence member's bbox and polygons.

    Input
    -----
    all_dets : list[dict]
        Each det: {"cls": int, "conf": float,
                   "xyxy": np.ndarray([x1,y1,x2,y2], float32),
                   "segs": list[list[float]]}  # polygons may be empty

    Output
    ------
    list[dict]
        [{"cls": int, "votes": int, "conf": float,
          "xyxy": list[float], "segs": list[list[float]]}, ...]
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


# -----------------------
# TIFF → fused images
# -----------------------
def build_fusions_for_tiff(tif_path, out_dir, rng=None):
    """
    Lit un TIFF, le redimensionne (comme au train), génère des fusions RGB
    pour toutes les combinaisons 1..3 canaux (par tranche Z si n>1),
    avec éventuels canaux "bruit" de la même tranche.

    Retourne: [(img_path, combo_tuple)], dims (H,W), (n,c)
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
    return fusions, (H, W), (n, c)



def coco_skeleton(categories):
    """Retourne un dict COCO vide avec la liste des catégories."""
    cats = [{"id": cid, "name": name, "supercategory": "hf"} 
            for name, cid in sorted(categories.items(), key=lambda x: x[1])]
    return {"images": [], "annotations": [], "categories": cats}

def poly_area_xy(poly):
    """Aire d’un polygone (liste [x1,y1,...]) en px^2 (via Shoelace)."""
    if not poly or len(poly) < 6: 
        return 0.0
    it = iter(poly)
    pts = [(float(x), float(next(it))) for x in it]
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    return 0.5 * abs(sum(x[i]*y[(i+1)%len(pts)] - x[(i+1)%len(pts)]*y[i] for i in range(len(pts))))

def bbox_xyxy_to_xywh(b):
    x1,y1,x2,y2 = map(float, b)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]



# -----------------------
# Main predict
# -----------------------
def run():
    """
    Predict from TIFFs with channel-fusion ensembling and vote consolidation.
    For each TIFF:
      - generate fused RGB images (all valid channel combos, same as training),
      - run YOLO on all fusions,
      - consolidate detections across fusions (IoU voting),
      - export both consolidated.json and coco.json (boxes + polygons).
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

    # Class names from YAML (training dataset YAML)
    yaml_path = HFinder_settings.get("yaml") or os.path.join(HFinder_folders.get_dataset_dir(), "dataset.yaml")
    if not os.path.isfile(yaml_path):
        HFinder_log.fail(f"dataset YAML not found: {yaml_path}")
    class_ids = HFinder_utils.load_class_definitions_from_yaml(yaml_path)  # {"name": id}

    overlay_iou = HFinder_settings.get("overlay_iou") or 0.5
    overlay_policy = (HFinder_settings.get("overlay_policy") or "keep_best").lower()
    try:
        wl_raw = HFinder_settings.get("overlay_whitelist") or "[]"
        wl_pairs = json.loads(wl_raw)  # e.g. [["haustoria","hyphae"]]
    except Exception:
        wl_pairs = []

    whitelist_ids = build_whitelist_ids(wl_pairs, class_ids)  # set of frozenset({idA,idB})

    if wl_pairs and not whitelist_ids:
        HFinder_log.warn("overlay_whitelist provided but no names matched class IDs from YAML.")

    HFinder_log.info(f"Overlay policy={overlay_policy}, IoU={overlay_iou}, whitelist_pairs={len(wl_pairs)}")

    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        torch.set_num_interop_threads(1)

    # Input TIFFs
    folder = HFinder_settings.get("tiff_dir")
    if not folder or not os.path.isdir(folder):
        HFinder_log.fail(f"Invalid 'tiff_dir': {folder}")
    tiffs = sorted(glob(os.path.join(folder, "*.tif")) + glob(os.path.join(folder, "*.tiff")))
    if not tiffs:
        HFinder_log.warn(f"No TIFF files found in {folder}")
        return

    project = HFinder_folders.get_runs_dir()
    HFinder_log.info(f"Predicting on {len(tiffs)} TIFF(s) | conf={conf} | imgsz={imgsz} | device={device}")

    # ---- TIFF loop -----------------------------------------------------------
    for tif_path in tiffs:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        out_dir = os.path.join(project, "predict", base)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Generate fused RGB images (same logic as training)
        rng = random.Random(base)  # deterministic per TIFF
        fusions, (H, W), (n, c) = build_fusions_for_tiff(tif_path, out_dir, rng=rng)
        fusion_paths = [p for p, _ in fusions]
        if not fusion_paths:
            HFinder_log.warn(f"[{base}] No fused images generated; skipping")
            continue
        HFinder_log.info(f"[{base}] Generated {len(fusion_paths)} fused images (n={n}, c={c})")

        # 2) Run predictions on all fusions
        results = model.predict(
            source=fusion_paths,
            batch=batch,
            save=True,                # saves overlays into runs/predict/<base>/
            project=project,
            name=f"predict/{base}",
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
            retina_masks=True         # better polygon quality
        )

        # Helpers for polygons
        def polys_from_xy(inst_xy):
            """
            inst_xy can be (N,2) or a list of (N,2).
            Return list of flattened polygons: [[x1,y1,...], ...]
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
            """Fallback: contours from binary mask i."""
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
                x1 = float(max(0.0, min(bb[0], W))); y1 = float(max(0.0, min(bb[1], H)))
                x2 = float(max(0.0, min(bb[2], W))); y2 = float(max(0.0, min(bb[3], H)))
                det = {"cls": int(cc), "conf": float(sc), "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32), "segs": []}

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
            iou_inter=float(overlay_iou),
            iou_intra=float(HFinder_settings.get("vote_iou") or 0.5),
            min_votes=int(HFinder_settings.get("vote_min") or 2),
            whitelist_ids=whitelist_ids,
            policy=overlay_policy
        )

        # 5) Save JSONs
        # 5a) consolidated.json
        summary = {
            "tiff": os.path.basename(tif_path),
            "img_size": [int(W), int(H)],
            "vote_iou": iou_vote,
            "vote_min": min_votes,
            "detections": consolidated  # [{cls, votes, conf_mean, xyxy, segs}]
        }
        with open(os.path.join(out_dir, "consolidated.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # 5b) coco.json
        coco = coco_skeleton(class_ids)  # {"images": [], "annotations": [], "categories":[...]}
        ref_img_rel = os.path.basename(fusion_paths[0])
        image_id = 1
        coco["images"].append({"id": image_id, "file_name": ref_img_rel, "width": int(W), "height": int(H)})

        def bbox_xyxy_to_xywh(b):
            x1, y1, x2, y2 = map(float, b)
            return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

        def poly_area_xy(poly):
            if not poly or len(poly) < 6:
                return 0.0
            it = iter(poly)
            pts = [(float(x), float(next(it))) for x in it]
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            return 0.5 * abs(sum(x[i]*y[(i+1)%len(pts)] - x[(i+1)%len(pts)]*y[i] for i in range(len(pts))))

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
                "segmentation": segmentation,  # list of [x1,y1,...]
                "iscrowd": 0,
                "confidence": round(det["conf_mean"], 4),
                "votes": int(det["votes"])
            })
            ann_id += 1

        with open(os.path.join(out_dir, "coco.json"), "w") as f:
            json.dump(coco, f, indent=2)

        # 6) Logs
        counts = Counter([d["cls"] for d in consolidated])
        HFinder_log.info(
            f"[{base}] Consolidated {sum(counts.values())} detections across "
            f"{len(set(counts.elements()))} classes; COCO & consolidated saved."
        )





