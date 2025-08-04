"""
hfinder_train — Utilities for training YOLO models and managing log redirection

This module provides helper functions to train YOLO models using Ultralytics' API,
while ensuring all standard output and error streams are redirected to a log file.
It also handles restoration of original outputs after training. It depends on 
external modules that manage folder structures, settings, and logging behavior.
"""

import os
from ultralytics import YOLO
import hfinder_log as HFinder_log
import hfinder_utils as HFinder_utils
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



def run(**kwargs):
    """
    Trains a YOLOv8 model using the Ultralytics API and project-specific settings.

    This function:
    - Retrieves the YOLO model path and training parameters from HFinder_settings.
    - Builds the appropriate dataset and output paths.
    - Redirects all output (stdout and stderr) to a log file.
    - Trains the model using `YOLO.train()`.
    - Restores original output streams at the end.

    Parameters:
        **kwargs: Keyword arguments passed to `YOLO.train()`.
    """
    yaml = os.path.join(HFinder_folders.get_dataset_dir(), "dataset.yaml")
    model = HFinder_settings.get("model")
    epochs = HFinder_settings.get("epochs")
    HFinder_log.info(f"Training YOLOv8 for {epochs} epochs with model {model}")
    log_path = os.path.join(HFinder_folders.get_log_dir(), "train.log")
    stdout_fd, stderr_fd = HFinder_utils.redirect_all_output(log_path)
    try:       
        yolo = YOLO(model)
        yolo.train(data=yaml,
                   project=HFinder_folders.get_runs_dir(),
                   epochs=epochs,
                   imgsz=HFinder_settings.get("target_size")[0],
                   verbose=False,
                   **kwargs)
    finally:
        HFinder_utils.restore_output(stdout_fd, stderr_fd)

