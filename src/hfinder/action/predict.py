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
- build_channel_jpegs_from_tiff(): Generate fused RGB images from a multichannel TIFF.
- consolidate_boxes_two_stage(): Apply cross-class filtering and intra-class consolidation.
- coco_skeleton(): Build an empty COCO structure with given categories.
"""


import os
import cv2
import json
import math
import torch
import tifffile
import numpy as np
from glob import glob
from collections import defaultdict, Counter
from ultralytics import YOLO

from hfinder.core import log as HF_log
from hfinder.core import utils as HF_utils
from hfinder.core import palette as HF_palette
from hfinder.core import geometry as HF_geometry
from hfinder.image import processing as HF_ImageOps
from hfinder.session import folders as HF_folders
from hfinder.session import settings as HF_settings



def resolve_device(raw):
    """
    Resolve a device string or integer to a PyTorch-compatible device.

    :param raw: Raw device specifier ("cpu", "auto", int, etc.).
    :type raw: str | int | None
    :return: "cpu" or CUDA device index.
    :rtype: str | int
    """
    if isinstance(raw, str):
        raw = raw.strip().lower()   
        if raw == "cpu":
            return "cpu"
        if raw in ("auto", ""):
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
    Map class-name pairs to order-agnostic class-ID pairs.
    Each valid name pair is converted to a frozenset of 
    two class IDs, so that (A, B) and (B, A) are treated 
    identically.

    :param whitelist_pairs_names: Iterable of class name pairs.
    :type whitelist_pairs_names: iterable[tuple[str, str] | list[str]]
    :param name_to_id: Mapping from class names to class IDs.
    :type name_to_id: dict[str, int]
    :return: Set of frozensets of class IDs.
    :rtype: set[frozenset[int]]
    """
    return {
        frozenset({name_to_id[a], name_to_id[b]})
        for a, b in whitelist_pairs_names
        if a in name_to_id and b in name_to_id
    }


def build_channel_jpegs_from_tiff(tif_path, out_dir):
    """
    Export each channel of a multichannel TIFF as a 
    JPEG frame. The TIFF is resized via 
    `HF_ImageOps.resize_multichannel_image()`, then
    each channel is saved as an RGB JPEG where 
    R = G = B (grayscale encoded in RGB).

    :param tif_path: Path to the input TIFF.
    :type tif_path: str
    :param out_dir: Output directory for JPEG frames.
    :type out_dir: str
    :return: (ratio, frames, (H, W), (n, c)) where `frames` is a list of dicts
             {"path": ..., "channel": ...}.
    :rtype: tuple[object, list[dict[str, object]], tuple[int, int], tuple[int, int]]
    """
    with tifffile.TiffFile(tif_path) as tf:
        series = tf.series[0]
        img = series.asarray()
        axes = series.axes
    
    # Note: in the current implementation, we always have n == 1
    channels_dict, ratio, (n, c) = HF_ImageOps.resize_multichannel_image(img, axes=axes)
    all_channels = sorted(channels_dict.keys())

    os.makedirs(out_dir, exist_ok=True)

    frames = []
    base = os.path.splitext(os.path.basename(tif_path))[0]

    for i in all_channels:
        out_path = os.path.join(out_dir, f"{base}_{i}.jpg")
        HF_ImageOps.save_gray_as_rgb(channels_dict[i], out_path)
        frames.append({
            "path": out_path,
            "channel": i,
        })

    H, W = next(iter(channels_dict.values())).shape
    return ratio, frames, (H, W), (n, c)



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



def channel_scores(det_subset):
    """
    Compute per-channel detection scores to identify the dominant class.

    Each detection contributes its confidence squared (conf²) to its class score.
    This non-linear weighting amplifies strong predictions while suppressing the
    influence of many weak ones.

    :param det_subset: Detections for one channel (each with keys 'cls' and 'conf').
    :type det_subset: list[dict]
    :return: (best_cls, best_cls_score, total_score, scores) where:
             - best_cls (int): class id with the highest cumulative score,
             - best_cls_score (float): score of that dominant class,
             - total_score (float): sum of all class scores for the channel,
             - scores (dict[int, float]): per-class score map (cls → Σ conf²).
    :rtype: tuple[int, float, float, dict[int, float]]
    """
    scores = defaultdict(float)
    method = HF_settings.get("power")

    if method == "log":
        # Pre-define a function for the log-likelihood scoring
        def scoring(x):
            return -math.log(1 - x + 1e-6)
    else:
        try:
            n = int(method)
        except Exception:
            HF_log.warn(f"Unknown method {method}")
            n = 4  # Fallback
        # Pre-define a function for power-based scoring
        def scoring(x, n=n):
            return x ** n

    for d in det_subset:
        c = int(d["cls"])
        x = float(d.get("conf", 0.0))
        scores[c] += scoring(x)

    if not scores:
        return -1, 0.0, 0.0, {}

    best_cls, best_cls_score = max(scores.items(), key=lambda kv: kv[1])
    total_score = sum(scores.values())
    return best_cls, best_cls_score, total_score, scores



def run():
    """
    Run full prediction pipeline:
      - Extract individual channels from TIFFs.
      - Perform YOLO predictions.
      - Consolidate detections.
      - Save outputs (consolidated.json, coco.json).

    :rtype: None
    """

    # Retrieve YOLO model.
    model_path = HF_settings.get("model")
    if not model_path:
        HF_log.fail("Model needed to perform predictions")
    if not os.path.exists(model_path):
        HF_log.fail(f"Model file not found: {model_path}")
    model = YOLO(model_path)

    # Inference settings (confidence, image size, and batch size)
    conf = HF_settings.get("confidence")
    imgsz = HF_settings.get("size")
    batch = HF_settings.get("batch")

    # Load (class id <-> name) mapping from the dataset YAML (kept stable across runs).
    yaml_path = HF_settings.get("yaml")
    if not os.path.isfile(yaml_path):
        HF_log.fail(f"YAML file not found: {yaml_path}")
    class_ids = HF_utils.load_class_definitions_from_yaml(yaml_path)
    id_to_name = {v: k for k, v in class_ids.items()}

    # Optional whitelist of class pairs allowed to be overlaid when co-occurring 
    # (JSON list of name pairs, e.g., [["haustoria", "hyphae"]]).
    try:
        wl_raw = HF_settings.get("overlay_whitelist") or "[]"
        wl_pairs = json.loads(wl_raw)
    except Exception:
        wl_pairs = []
    whitelist_ids = build_whitelist_ids(wl_pairs, class_ids)  # set of frozenset({idA,idB})
    if wl_pairs and not whitelist_ids:
        HF_log.warn("Overlay whitelist provided but no names matched class IDs from YAML.")

    # On CPU, limit PyTorch thread parallelism to reduce oversubscription.
    device = resolve_device(HF_settings.get("device"))
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        torch.set_num_interop_threads(1)

    # Input TIFFs
    input_folder = HF_settings.get("tiff_dir")
    if not input_folder or not os.path.isdir(input_folder):
        HF_log.fail(f"Invalid directory: {input_folder}")
    tiffs = sorted(glob(os.path.join(input_folder, "*.tif")) + \
                   glob(os.path.join(input_folder, "*.tiff")))
    if not tiffs:
        HF_log.warn(f"No TIFF files found in {input_folder}")
        return

    project = HF_folders.get_runs_dir()
    HF_log.info(f"Predicting on {len(tiffs)} TIFF(s) | conf={conf} | imgsz={imgsz} | device={device}")

    for tif_path in tiffs:
        tif_file = os.path.basename(tif_path)
        tif_base = os.path.splitext(tif_file)[0]
        out_dir = os.path.join(project, "predict", tif_base)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Generate fused RGB images (same logic as training)
        ratio, frames, (H, W), (n, c) = build_channel_jpegs_from_tiff(tif_path, out_dir)
        orig_W = int(round(W / ratio))
        orig_H = int(round(H / ratio))
        scale_factor = 1.0 / ratio
        fusion_paths = [f["path"] for f in frames]
        fusion_meta = {f["path"]: f for f in frames}
        if not fusion_paths:
            HF_log.warn(f"Skipping file '{tif_file}'")
            continue
        HF_log.info(f"Processing '{tif_file}'")

        # 2) Run predictions on all frames
        results = model.predict(
            source=fusion_paths,
            batch=batch,
            save=True,            # saves overlays into runs/predict/<tif_base>/
            iou=0.5,
            project=project,
            name=os.path.join("predict", tif_base),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
            retina_masks=True     # better polygon quality
        )

        # 3) Collect detections (boxes + polygons)
        all_dets = []
        for img_path, r in zip(fusion_paths, results):
            if getattr(r, "boxes", None) is None or r.boxes is None or r.boxes.shape[0] == 0:
                continue
            meta = fusion_meta.get(img_path, {})
            channel = meta.get("channel")
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
                    "segs": [],
                    "channel": channel
                }

                # 1) Try masks.xy (handles (N,2) OR list of (N,2))
                if xy_polys is not None and i < len(xy_polys) and xy_polys[i] is not None:
                    det["segs"] = HF_geometry.polys_from_xy(xy_polys[i])

                # 2) Fallback: contours from binary mask
                if not det["segs"]:
                    det["segs"] = polys_from_mask_i(r, i)

                all_dets.append(det)

        # Generating detection subsets for each channel
        subsets = []
        for ch in range(len(frames)):
            det_subset = [det for det in all_dets if det["channel"] == ch]
            if not det_subset:
                continue
            subsets.append((ch, det_subset))
       
        # Sorting subsets by detection score:
        subset_info = []
        for ch, det_subset in subsets:
            best_cls, best_cls_score, total_score, scores = channel_scores(det_subset)
            subset_info.append((ch, det_subset, best_cls, best_cls_score, total_score, scores))

        # Sort by (best_cls_score, total_score, count)
        subset_info.sort(key=lambda t: (t[3], t[4], len(t[1])), reverse=True)
       
        # ---- COCO skeleton
        coco = coco_skeleton(class_ids)
        image_id = 1
        coco["images"].append({
            "id": image_id,
            "file_name": tif_file,
            "width":  int(W * scale_factor),
            "height": int(H * scale_factor),
        })
        ann_id = 1  # global per TIFF

        already_assigned = set()
        for ch, subset, best, best_score, total_score, scores in subset_info:
            if not scores:
                continue

            present = set(scores.keys())

            if best in already_assigned:
                pairs = [
                    (c, a) for c in present for a in present
                    if c != a and frozenset({c, a}) in whitelist_ids
                ]
                if not pairs:
                    continue
                c_star, a_star = max(pairs, key=lambda p: scores[p[0]] + scores[p[1]])
                kept_set = {c_star, a_star}
            else:
                partners = {c for c in present if frozenset({c, best}) in whitelist_ids}
                partners &= present
                kept_set = {best} | (partners if partners else set())

            already_assigned |= kept_set

            subset = [d for d in subset if d["cls"] in kept_set]
            if not subset:
                continue

            filtered = [
                d for d in subset
                if (d["cls"] == best) or (frozenset({d["cls"], best}) in whitelist_ids)
            ]
            if not filtered:
                continue

            # Rescale
            rescaled = []
            for d in filtered:
                rescaled.append({
                    "cls": int(d["cls"]),
                    "conf": float(d["conf"]),
                    "xyxy": HF_geometry.rescale_box_xyxy(d["xyxy"], scale_factor),
                    "segs": [HF_geometry.rescale_seg_flat(seg, scale_factor) for seg in (d.get("segs") or [])],
                })

            for d in rescaled:
                bbox_xywh = HF_geometry.bbox_xyxy_to_xywh(d["xyxy"])
                if d["segs"]:
                    area = float(sum(HF_geometry.poly_area_xy(seg) for seg in d["segs"]))
                    segmentation = d["segs"]
                else:
                    area = float(bbox_xywh[2] * bbox_xywh[3])
                    segmentation = []

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": d["cls"],
                    "bbox": [round(v, 2) for v in bbox_xywh],
                    "area": round(area, 2),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                    "confidence": round(d["conf"], 4),
                    "hf_channel": ch, # ch ici = canal du subset_info (OK, 1-based si tes frames le sont)
                })
                ann_id += 1

        out_name = f"{tif_base}.json"
        with open(os.path.join(input_folder, out_name), "w") as f:
            json.dump(coco, f, indent=2)       
