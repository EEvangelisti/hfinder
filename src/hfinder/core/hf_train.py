"""
Training utilities for HFinder (YOLO via Ultralytics).

This module provides a thin wrapper around Ultralytics' YOLO training API to:
- Resolve project paths and settings (dataset YAML, runs dir, image size, epochs).
- Redirect stdout/stderr to a session log during training.
- Restore original output streams afterward.

Public API
----------
- run(**kwargs): Train a YOLO model using settings from HF_settings, writing
  logs to the current session's log directory.

Notes
-----
- Output redirection is handled by hf_utils.redirect_all_output() /
  hf_utils.restore_output(). If training raises, the finally block restores
  the original file descriptors.
"""

import gc
import os
import torch
from ultralytics import YOLO
from hfinder.core import log as HF_log
from hfinder.core import hf_utils as HF_utils
from hfinder.session import folders as HF_folders
from hfinder.session import settings as HF_settings



def run(**kwargs):
    """
    Train a YOLOv8 model using Ultralytics with project-specific settings.

    This function:
      - Retrieves the model path, epochs, and image size from HF_settings.
      - Resolves dataset YAML and output directories via HF_folders.
      - Redirects all output (stdout and stderr) to a session log file.
      - Invokes `YOLO.train()` with defaults and user-provided overrides.
      - Restores original output streams on exit.

    :param kwargs: Extra keyword arguments forwarded to `YOLO.train()`
        (e.g., batch, lr0, optimizer, device, workers, seed, etc.).
    :type kwargs: dict
    :return: None
    :rtype: None
    """
    
    # Resolve paths and settings.
    yaml = os.path.join(HF_folders.get_dataset_dir(), "dataset.yaml")
    model = HF_settings.get("model")
    epochs = HF_settings.get("epochs")
    HF_log.info(f"Training YOLOv8 for {epochs} epochs with model {model}")
    
    # Redirect all output to a session log file.
    log_path = os.path.join(HF_folders.get_log_dir(), "train.log")
    stdout_fd, stderr_fd = HF_utils.redirect_all_output(log_path)
    
    # Free up memory
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # Reset peak stats (PyTorch ≥1.9). Fallback for older versions.
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            try:
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()
            except Exception:
                pass
    
    try:
        # Initialize model and launch training.       
        yolo = YOLO(model)
        yolo.train(data=yaml,
                   project=HF_folders.get_runs_dir(),
                   epochs=epochs,
                   imgsz=HF_settings.get("size"),
                   verbose=False,
                   **kwargs)
    finally:
        # Ensure file descriptors are restored even if training fails.
        HF_utils.restore_output(stdout_fd, stderr_fd)

