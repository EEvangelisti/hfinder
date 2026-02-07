"""
Prediction pipeline for multichannel TIFF images.

This module implements the prediction stage of HFinder.
Multichannel TIFF files are decomposed into per-channel RGB JPEG images
(grayscale encoded as R=G=B). Object detection is performed independently
on each channel, and detections are then consolidated using class-level
competition and optional co-occurrence rules.

Key steps
---------
- Load multichannel TIFF images.
- Export one RGB JPEG per channel.
- Run YOLO predictions on each channel independently.
- Rank channels using confidence-based scoring.
- Enforce cross-class exclusivity, with optional whitelist-based co-occurrence.
- Export consolidated results in COCO-compatible JSON format.

Public API
----------
- run(): Run predictions from TIFFs and export results.
- resolve_device(raw): Normalize user-specified device identifiers.
- build_channel_jpegs_from_tiff(): Export per-channel JPEGs from a TIFF.
- build_whitelist_ids(): Build allowed class co-occurrence rules.
- coco_skeleton(): Build an empty COCO annotation structure.
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



def load_yolo_model():
    """
    Load the YOLO model specified in HFinder settings.

    This helper validates the configured model path (HF_settings["model"])
    and instantiates an ultralytics.YOLO object.

    :return: Loaded YOLO model ready for inference.
    :rtype: ultralytics.YOLO
    :raises SystemExit: via HF_log.fail() if the model path is missing or invalid.
    """
    model_path = HF_settings.get("model")
    if not model_path:
        HF_log.fail("Model needed to perform predictions")
    if not os.path.exists(model_path):
        HF_log.fail(f"Model file not found: {model_path}")
    return YOLO(model_path)


def prepare_frames_for_tiff(tif_path, tif_file, project, tif_base):
    """
    Prepare per-channel JPEG frames for a multichannel TIFF.

    The TIFF is resized (via HF_ImageOps.resize_multichannel_image) and each channel
    is exported as an RGB JPEG where R=G=B (grayscale encoded as RGB). The returned
    `fusion_paths` are the inputs expected by YOLO inference.

    Notes
    -----
    Despite historical naming, no channel "fusion" is performed here: one JPEG is
    generated per channel.

    :param tif_path: Path to the input TIFF file.
    :type tif_path: str
    :param project: Root run directory (HF_folders.get_runs_dir()).
    :type project: str
    :param tif_base: Basename used to create the output folder under runs/predict/.
    :type tif_base: str
    :param tif_file: Optional TIFF filename for logging (defaults to basename(tif_path)).
    :type tif_file: str | None
    :return: (out_dir, ratio, frames, H, W, scale_factor, fusion_paths, fusion_meta)
             where:
               - ratio is the resize ratio returned by resize_multichannel_image,
               - (H, W) are the resized frame dimensions,
               - scale_factor = 1/ratio rescales predictions back to original size,
               - frames is a list of dicts {"path": ..., "channel": ...},
               - fusion_paths is the list of JPEG paths (one per channel),
               - fusion_meta maps JPEG path -> frame dict.
    :rtype: tuple[str, object, list[dict], int, int, float, list[str], dict[str, dict]]
    """
    out_dir = os.path.join(project, "predict", tif_base)
    os.makedirs(out_dir, exist_ok=True)
    ratio, frames, (H, W), (n, c) = build_channel_jpegs_from_tiff(tif_path, out_dir)
    orig_W = int(round(W / ratio))
    orig_H = int(round(H / ratio))
    scale_factor = 1.0 / ratio
    fusion_paths = [f["path"] for f in frames]
    fusion_meta = {f["path"]: f for f in frames}
    HF_log.info(f"Processing '{tif_file}'")
    return out_dir, ratio, frames, H, W, scale_factor, fusion_paths, fusion_meta


def predict_on_frames(model, fusion_paths, project, tif_base, conf, imgsz, batch, device):
    """
    Run YOLO inference on a list of per-channel JPEG frames.

    This is a thin wrapper around `ultralytics.YOLO.predict()` configured to:
      - process all channel frames for a given TIFF,
      - save overlay images under runs/predict/<tif_base>/ (save=True),
      - return per-frame Results objects (boxes + optional masks).

    :param model: Loaded ultralytics YOLO model.
    :type model: ultralytics.YOLO
    :param fusion_paths: List of JPEG paths (one per channel).
    :type fusion_paths: list[str]
    :param project: Root run directory (ultralytics "project" argument).
    :type project: str
    :param tif_base: Subdirectory name under project/predict/.
    :type tif_base: str
    :param conf: Confidence threshold for inference.
    :type conf: float
    :param imgsz: Inference image size.
    :type imgsz: int | tuple[int, int]
    :param batch: Batch size for inference.
    :type batch: int
    :param device: Device specifier ("cpu" or CUDA index).
    :type device: str | int
    :return: One ultralytics Results object per input frame.
    :rtype: list[ultralytics.engine.results.Results]
    """
    return model.predict(
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


def collect_detections(results, fusion_paths, fusion_meta, W, H):
    """
    Convert YOLO Results into a flat list of detection dicts annotated with channel id.

    For each predicted instance, this function:
      - clamps the bounding box to the resized frame bounds (W, H),
      - extracts polygons either from YOLO masks (preferred) or from the binary mask
        via OpenCV contours (fallback),
      - records the channel index associated with the source frame.

    The output format is designed to be consumed by the channel attribution /
    consolidation stage (semantic selection across confocal channels).

    :param results: YOLO results aligned with `fusion_paths`.
    :type results: list[ultralytics.engine.results.Results]
    :param fusion_paths: List of frame paths used for inference.
    :type fusion_paths: list[str]
    :param fusion_meta: Mapping frame path -> {"path": ..., "channel": ...}.
    :type fusion_meta: dict[str, dict]
    :param W: Resized frame width.
    :type W: int
    :param H: Resized frame height.
    :type H: int
    :return: List of detections with keys: cls, conf, xyxy, segs, channel.
    :rtype: list[dict]
    """
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

            # Try masks.xy (handles (N,2) OR list of (N,2))
            if xy_polys is not None and i < len(xy_polys) and xy_polys[i] is not None:
                det["segs"] = HF_geometry.polys_from_xy(xy_polys[i])

            # Fallback: contours from binary mask
            if not det["segs"]:
                det["segs"] = polys_from_mask_i(r, i)

            all_dets.append(det)

    return all_dets


def consolidate_to_coco(all_dets, frames, class_ids, whitelist_ids, tif_file, W, H, scale_factor):
    """
    Consolidate per-channel detections into a single COCO annotation dict for one TIFF.

    Rationale (confocal prior)
    --------------------------
    In confocal microscopy, each channel corresponds to one signal (fluorophore) and
    typically to one target class, possibly with a small set of whitelist-approved
    co-occurring classes. Therefore, consolidation is performed semantically:
    channels are ranked by class evidence and only the most plausible channel/class
    assignments are kept.

    Procedure
    ---------
    1) Group detections by channel.
    2) For each channel, compute per-class cumulative scores and select the dominant class.
    3) Rank channels by (dominant class score, total score, detection count).
    4) Traverse channels from strongest to weakest and keep:
         - the dominant class if not already claimed,
         - optional whitelist partners,
         - or (if the dominant class is already claimed) only a whitelist-approved pair.
    5) Rescale boxes/polygons back to the original TIFF scale and emit COCO annotations.

    Guardrail (recommended, not implemented here)
    ---------------------------------------------
    Bleed-through and autofluorescence can create an overconfident but wrong dominant class.
    A robust guardrail is to require a minimum dominance ratio per channel:

        dominance = best_cls_score / total_score

    Channels with dominance below a threshold (e.g. 0.6) can be rejected as ambiguous.

    :param all_dets: Flat list of detection dicts produced by collect_detections().
    :type all_dets: list[dict]
    :param frames: Frame descriptors produced by prepare_frames_for_tiff().
    :type frames: list[dict]
    :param class_ids: Mapping class name -> class id.
    :type class_ids: dict[str, int]
    :param whitelist_ids: Set of allowed unordered class-id pairs (frozenset({a,b})).
    :type whitelist_ids: set[frozenset[int]]
    :param tif_file: TIFF filename (for COCO 'file_name').
    :type tif_file: str
    :param W: Resized frame width.
    :type W: int
    :param H: Resized frame height.
    :type H: int
    :param scale_factor: Factor applied to map resized coordinates back to original size.
    :type scale_factor: float
    :return: COCO-like dict with one image entry and consolidated annotations.
    :rtype: dict
    """
    # a) Build detection subsets per channel (skip empty channels).
    subsets = []
    for ch in range(len(frames)):
        det_subset = [det for det in all_dets if det["channel"] == ch]
        if not det_subset:
            continue
        subsets.append((ch, det_subset))
    
    # b) Score each channel: compute per-class cumulative score and dominant class.
    # channel_scores() returns:
    #   - best_cls: argmax_c sum(conf^n) (or log scoring)
    #   - best_cls_score: that max score
    #   - total_score: sum over classes
    #   - scores: per-class score map
    subset_info = []
    for ch, det_subset in subsets:
        best_cls, best_cls_score, total_score, scores = channel_scores(det_subset)
        # Guardrail (NOT IMPLEMENTED): reject ambiguous channels.
        # dominance = best_cls_score / max(total_score, 1e-12)
        # if dominance < DOMINANCE_MIN:
        #     continue
        subset_info.append((ch, det_subset, best_cls, best_cls_score, total_score, scores))
    
    # c) Rank channels from strongest to weakest.
    # Primary: dominant class evidence, Secondary: overall evidence, Tertiary: detection count.
    subset_info.sort(key=lambda t: (t[3], t[4], len(t[1])), reverse=True)
    
    # COCO skeleton
    coco = coco_skeleton(class_ids)
    image_id = 1
    coco["images"].append({
        "id": image_id,
        "file_name": tif_file,
        "width":  int(W * scale_factor),
        "height": int(H * scale_factor),
    })
    ann_id = 1  # global per TIFF
    
    # d) Traverse channels in ranked order and decide which classes are allowed to survive.
    # already_assigned tracks dominant/kept classes already claimed by stronger channels.
    already_assigned = set()
    for ch, subset, best, best_score, total_score, scores in subset_info:
        if not scores:
            continue
    
        # Classes predicted in this channel.
        present = set(scores.keys())
    
        # If this channel's dominant class was already claimed by a stronger channel,
        # we only keep this channel if it supports a whitelist-approved co-occurrence
        # (i.e., two classes allowed to overlay in the same acquisition context).
        # Otherwise, we keep the dominant class and optionally add whitelist partners
        # (classes allowed to co-occur with the dominant one).
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
                "hf_channel": ch
            })
            ann_id += 1

    return coco


def run():
    """
    Run the HFinder prediction pipeline on a directory of multichannel TIFF files.

    Overview
    --------
    For each TIFF:
      1) Export one RGB JPEG per channel (grayscale encoded as R=G=B) after resizing.
      2) Run YOLO inference independently on each channel frame.
      3) Attribute detections to channels and consolidate them across channels using
         class-level competition and optional whitelist-based co-occurrence rules.
      4) Export a single COCO JSON (one COCO image per TIFF) with consolidated annotations.

    Channel attribution and consolidation rationale
    -----------------------------------------------
    Confocal imaging provides a strong prior: one channel corresponds to one signal
    (one fluorophore), hence typically to one class, or to a small set of allowed
    co-occurring classes. Therefore, consolidation is performed *semantically* rather
    than geometrically (no multi-channel IoU voting / NMS).

    The procedure is:
      - Group detections by channel.
      - For each channel, compute per-class scores (default: sum(conf^n), or log scoring),
        and identify the dominant class ("best").
      - Rank channels by dominance (best class score, then total score, then count).
      - Traverse channels from strongest to weakest and decide which classes to keep:
          * If the dominant class was not assigned yet, keep it (and optionally its
            whitelist partners).
          * If the dominant class was already assigned by a stronger channel, only keep
            this channel if it supports an allowed co-occurring pair (via whitelist).
      - Emit COCO annotations from the retained detections, rescaled back to original size,
        and store the chosen channel as metadata (hf_channel).

    Safety guardrail (not implemented yet)
    --------------------------------------
    A common confocal failure mode is bleed-through/autofluorescence leading to an
    overconfident but wrong "dominant" class. A simple guardrail is to require a minimum
    dominance ratio per channel:

        dominance = best_cls_score / total_score

    If dominance is below a threshold (e.g. 0.6), the channel is considered ambiguous
    and can be skipped or down-weighted. This prevents a noisy channel with scattered
    detections from dominating the consolidation.

    Outputs
    -------
    - Writes <tif_base>.json in the input TIFF directory (COCO-like structure).
    - YOLO overlays are saved under the run folder (ultralytics 'save=True').

    :rtype: None
    """

    model = load_yolo_model()

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
    whitelist_ids = build_whitelist_ids(wl_pairs, class_ids)
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
        out_dir, ratio, frames, H, W, scale_factor, fusion_paths, fusion_meta = prepare_frames_for_tiff(tif_path, tif_file, project, tif_base) 
        if not fusion_paths:
            HF_log.warn(f"Skipping file '{tif_file}'")
            continue
        # Run predictions on all frames
        results = predict_on_frames(model, fusion_paths, project, tif_base, conf, imgsz, batch, device)
        # Collect detections (boxes + polygons)
        all_dets = collect_detections(results, fusion_paths, fusion_meta, W, H)
        # Consildate results and generate the output COCO file
        coco = consolidate_to_coco(all_dets, frames, class_ids, whitelist_ids, tif_file, W, H, scale_factor)
        # Save the output COCO file
        out_name = f"{tif_base}.json"
        with open(os.path.join(input_folder, out_name), "w") as f:
            json.dump(coco, f, indent=2)       
