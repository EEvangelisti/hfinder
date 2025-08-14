# Preliminary prediction pipeline (minimal but robust)

import os
from glob import glob
from ultralytics import YOLO
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



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
    device = HFinder_settings.get("device") or 0  # "cpu" ou index GPU
    # Dossier de sortie homogène avec l'entraînement
    project = HFinder_folders.get_runs_dir()
    name = "predict"  # sous-dossier

    HFinder_log.info(
        f"Predicting {len(images)} image(s) | conf={conf} | imgsz={imgsz} | device={device}"
    )

    for image_path in images:
        results = model.predict(
            source=image_path,
            save=True,
            project=project,
            name=name,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        # results est une liste de 'Results' (en général taille 1 pour source=chemin unique)
        for r in results:
            n_boxes = int(r.boxes.shape[0]) if (r.boxes is not None and r.boxes.shape[0] > 0) else 0
            n_masks = int(r.masks.data.shape[0]) if (getattr(r, "masks", None) is not None and r.masks.data is not None) else 0
            img_name = os.path.basename(image_path)
            HFinder_log.info(f"Image {img_name}: {n_boxes} boxes, {n_masks} masks")
