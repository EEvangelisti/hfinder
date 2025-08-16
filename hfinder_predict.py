# Preliminary prediction pipeline (minimal but robust)

import os
import torch
from glob import glob
from ultralytics import YOLO
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



def resolve_device(raw):
    # Cas explicites
    if isinstance(raw, str) and raw.strip().lower() == "cpu":
        return "cpu"
    if isinstance(raw, str) and raw.strip().lower() in ("auto", ""):
        raw = None

    # Index numérique éventuel
    if raw is None:
        return 0 if torch.cuda.is_available() else "cpu"
    try:
        idx = int(raw)
        # Si Torch ne voit pas de GPU, on retombe sur CPU
        return idx if torch.cuda.is_available() and (0 <= idx < torch.cuda.device_count()) else "cpu"
    except (TypeError, ValueError):
        # Valeur non comprise → CPU par sécurité
        return "cpu"



def run():
    weights = HFinder_settings.get("weights")
    if not weights:
        HFinder_log.fail("Weights needed to perform predictions")
    if not os.path.exists(weights):
        HFinder_log.fail(f"Weights file not found: {weights}")

    model = YOLO(weights)

    folder = HFinder_settings.get("tiff_dir")
    if not folder or not os.path.isdir(folder):
        HFinder_log.fail(f"Invalid 'tiff_dir': {folder}")

    # TODO: Move to TIFF
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for pat in patterns:
        images.extend(glob(os.path.join(folder, pat)))
    images = sorted(images)

    if not images:
        HFinder_log.warn(f"No images found in {folder} (patterns: {patterns})")
        return

    # 3) Réglages d'inférence
    conf = HFinder_settings.get("conf") or 0.25
    imgsz = HFinder_settings.get("size") or 640
    device = resolve_device(HFinder_settings.get("device"))
    
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        torch.set_num_interop_threads(1)
    
    # Dossier de sortie homogène avec l'entraînement
    project = HFinder_folders.get_runs_dir()
    name = "predict"  # sous-dossier

    HFinder_log.info(
        f"Predicting {len(images)} image(s) | conf={conf} | imgsz={imgsz} | device={device}"
    )

    results_list = model.predict(
        source=images,
        batch=8,
        save=True,
        project=project,
        name=name,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False
    )

    # results est une liste de 'Results' (en général taille 1 pour source=chemin unique)
    for img_path, r in zip(images, results_list):
        n_boxes = int(r.boxes.shape[0]) if (getattr(r, "boxes", None) is not None and r.boxes is not None) else 0
        n_masks = 0
        if getattr(r, "masks", None) is not None and getattr(r.masks, "data", None) is not None:
            n_masks = int(getattr(r.masks.data, "shape", [0])[0])
        HFinder_log.info(f"Image {os.path.basename(img_path)}: {n_boxes} boxes, {n_masks} masks")
