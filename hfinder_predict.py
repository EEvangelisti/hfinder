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
def iou_xyxy(a, b):
    """IoU entre deux boîtes xyxy (np.array [x1,y1,x2,y2])."""
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    denom = area_a + area_b - inter + 1e-9
    return inter / denom


def consolidate_boxes(all_dets, iou_thresh=0.5, min_votes=2):
    """
    all_dets: liste de dicts
      {"cls": int, "conf": float, "xyxy": np.array([x1,y1,x2,y2]),
       "segs": [ [x1,y1,...], [x1,y1,...], ... ]  # peut être vide }
    """
    by_class = defaultdict(list)
    for d in all_dets:
        by_class[d["cls"]].append(d)

    consolidated = []
    for cls, dets in by_class.items():
        clusters = []  # {members, xyxy, conf_sum, best_any, best_with_segs}
        for det in dets:
            matched = False
            for cl in clusters:
                if iou_xyxy(det["xyxy"], cl["xyxy"]) >= iou_thresh:
                    cl["members"].append(det)
                    # mise à jour barycentre pondéré par la confiance
                    wsum = cl["conf_sum"] + det["conf"]
                    cl["xyxy"] = (cl["xyxy"] * cl["conf_sum"] + det["xyxy"] * det["conf"]) / max(1e-9, wsum)
                    cl["conf_sum"] = wsum
                    # meilleur global
                    if det["conf"] > cl["best_any"]["conf"]:
                        cl["best_any"] = det
                    # meilleur avec polygones
                    if det.get("segs") and (not cl["best_with_segs"] or det["conf"] > cl["best_with_segs"]["conf"]):
                        cl["best_with_segs"] = det
                    matched = True
                    break
            if not matched:
                clusters.append({
                    "members": [det],
                    "xyxy": det["xyxy"].copy(),
                    "conf_sum": det["conf"],
                    "best_any": det,
                    "best_with_segs": det if det.get("segs") else None,
                })

        for cl in clusters:
            votes = len(cl["members"])
            if votes >= min_votes:
                mean_conf = cl["conf_sum"] / votes
                # priorité au meilleur qui a des polygones
                chosen = cl["best_with_segs"] if cl["best_with_segs"] is not None else cl["best_any"]
                segs = chosen.get("segs") or []  # peut rester vide si aucun n'avait de polygone
                consolidated.append({
                    "cls": cls,
                    "votes": votes,
                    "conf_mean": float(mean_conf),
                    "xyxy": cl["xyxy"].tolist(),
                    "segs": segs
                })
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
    Ensembling sur TIFF : génère des fusions RGB pour toutes combinaisons 1..3 canaux,
    prédit avec YOLO, puis consolide par vote/IoU les détections récurrentes.
    Écrit consolidated.json et coco.json par TIFF sous runs/predict/<TIFF_BASE>/.
    """
    # Poids / modèle
    weights = HFinder_settings.get("weights")
    if not weights:
        HFinder_log.fail("Weights needed to perform predictions")
    if not os.path.exists(weights):
        HFinder_log.fail(f"Weights file not found: {weights}")
    model = YOLO(weights)

    # Réglages
    conf = HFinder_settings.get("conf") or 0.25
    imgsz = HFinder_settings.get("size") or 640
    device = resolve_device(HFinder_settings.get("device"))
    batch = HFinder_settings.get("batch") or 8
    iou_vote = float(HFinder_settings.get("vote_iou") or 0.5)
    min_votes = int(HFinder_settings.get("vote_min") or 2)

    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        torch.set_num_interop_threads(1)

    # Entrée TIFFs
    folder = HFinder_settings.get("tiff_dir")
    if not folder or not os.path.isdir(folder):
        HFinder_log.fail(f"Invalid 'tiff_dir': {folder}")
    tiffs = sorted(glob(os.path.join(folder, "*.tif")) + glob(os.path.join(folder, "*.tiff")))
    if not tiffs:
        HFinder_log.warn(f"No TIFF files found in {folder}")
        return

    project = HFinder_folders.get_runs_dir()
    HFinder_log.info(
        f"Predicting on {len(tiffs)} TIFF(s) | conf={conf} | imgsz={imgsz} | device={device}"
    )

    # Boucle TIFF par TIFF pour consolider séparément
    for tif_path in tiffs:
        base = os.path.splitext(os.path.basename(tif_path))[0]
        out_dir = os.path.join(project, "predict", base)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Générer les fusions RGB locales
        rng = random.Random(base)  # déterministe par TIFF
        fusions, (H, W), (n, c) = build_fusions_for_tiff(tif_path, out_dir, rng=rng)
        fusion_paths = [p for p, _ in fusions]
        if not fusion_paths:
            HFinder_log.warn(f"[{base}] No fused images generated; skipping")
            continue

        HFinder_log.info(f"[{base}] Generated {len(fusion_paths)} fused images (n={n}, c={c})")

        # 2) Prédire sur toutes les fusions (Ultralytics gère la liste de chemins)
        results = model.predict(
            source=fusion_paths,
            batch=batch,
            save=True,           # enregistre overlays dans out_dir
            project=project,
            name=f"predict/{base}",
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
            retina_masks=True
        )

        # 3) Collecter toutes les boxes + polygones pour vote IoU
        all_dets = []  # liste de dicts {"cls","conf","xyxy","segs"}
        for img_path, r in zip(fusion_paths, results):
            if getattr(r, "boxes", None) is None or r.boxes is None or r.boxes.shape[0] == 0:
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)

            # Polygones : r.masks.xy est une liste par instance (liste de polylignes)
            segs_list = None
            if getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None:
                segs_list = r.masks.xy  # liste de listes d’array (N_i x 2) en pixels

            for i, (bb, sc, cc) in enumerate(zip(xyxy, confs, clss)):
                # clamp dans l'espace des fusions (W,H) — tes fusions sont carrées si size carré
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
                if segs_list is not None and i < len(segs_list) and segs_list[i] is not None:
                    flats = []
                    for poly in segs_list[i]:
                        if poly is None or len(poly) < 3:
                            continue
                        flats.append(poly.reshape(-1).tolist())  # [x1,y1,...]
                    det["segs"] = flats
                all_dets.append(det)

        # 4) Consolidation
        consolidated = consolidate_boxes(all_dets, iou_thresh=iou_vote, min_votes=min_votes)

        # 5) Sauvegardes JSON
        # 5a) Résumé consolidated.json
        summary = {
            "tiff": os.path.basename(tif_path),
            "img_size": [int(W), int(H)],
            "vote_iou": iou_vote,
            "vote_min": min_votes,
            "detections": consolidated  # [{cls, votes, conf_mean, xyxy, segs}]
        }
        with open(os.path.join(out_dir, "consolidated.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # 5b) COCO coco.json
        # Prefer the exact dataset.yaml used for training
        dataset_yaml = HFinder_settings.get("yaml")
        if not os.path.isfile(dataset_yaml):
            HFinder_log.fail(f"dataset.yaml not found: {dataset_yaml}")

        class_ids = HFinder_utils.load_class_definitions_from_yaml(dataset_yaml)
        coco = coco_skeleton(class_ids)

        # on référence une image de fond (la première fusion)
        ref_img_rel = os.path.basename(fusion_paths[0])
        image_id = 1
        coco["images"].append({
            "id": image_id,
            "file_name": ref_img_rel,
            "width": int(W),
            "height": int(H),
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
                "segmentation": segmentation,  # liste de listes [x1,y1,...]
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
