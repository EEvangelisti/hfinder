"""
Training utilities for HFinder.

A thin wrapper around Ultralytics' YOLO training API to:
- Resolve project paths and settings (dataset YAML, runs dir, image size, epochs).
- Train a YOLO model with project defaults and user-provided overrides.

Public API
----------
- run(**kwargs): Train a YOLO model using settings from HF_settings.
"""

import gc
import os
import torch
from ultralytics import YOLO
from hfinder.core import log as HF_log
from hfinder.session import folders as HF_folders
from hfinder.session import settings as HF_settings



def run(**kwargs):
    """
    Train a YOLOv8 model using Ultralytics with project-specific settings.

    This function:
      - Retrieves the model path, epochs, and image size from HF_settings.
      - Resolves dataset YAML and output directories via HF_folders.
      - Invokes `YOLO.train()` with defaults and user-provided overrides.

    :param kwargs: Extra keyword arguments forwarded to `YOLO.train()`
        (e.g., batch, lr0, optimizer, device, workers, seed, etc.).
    :type kwargs: dict
    :return: None
    :rtype: None
    """
    
    # Resolve dataset YAML path.
    yaml = os.path.join(HF_folders.get_dataset_dir(), "dataset.yaml")
    
    model = HF_settings.get("model")
    epochs = HF_settings.get("epochs")
    imgsz = HF_settings.get("size")
    HF_log.info(f"Training YOLOv8 for {epochs} epochs with model {model} (imgsz={imgsz})")
    
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    yolo = YOLO(model)
    yolo.train(
        data=yaml,
        project=HF_folders.get_runs_dir(),
        epochs=epochs,
        imgsz=imgsz,
        verbose=False,
        **kwargs
    )
