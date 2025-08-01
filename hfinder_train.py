"""
hfinder_train — Utilities for training YOLO models and managing log redirection

This module provides helper functions to train YOLO models using Ultralytics' API,
while ensuring all standard output and error streams are redirected to a log file.
It also handles restoration of original outputs after training. It depends on 
external modules that manage folder structures, settings, and logging behavior.
"""

import os
import sys
import yaml
from ultralytics import YOLO
import hfinder_log as HFinder_log 
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



def redirect_all_output(log_path):
    """
    Redirects both stdout and stderr to a specified log file.

    Parameters:
        log_path (str): Path to the log file where output will be written.

    Returns:
        tuple: A pair of file descriptors (stdout_fd, stderr_fd) representing
               the original stdout and stderr. These must be passed to 
               `restore_output` to revert redirection.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    
    # Save original stdout and stderr file descriptors
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)

    # Redirect stdout and stderr to log_fd
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    return stdout_fd, stderr_fd



def restore_output(stdout_fd, stderr_fd):
    """
    Restores stdout and stderr to their original state.

    Parameters:
        stdout_fd (int): File descriptor of the original stdout.
        stderr_fd (int): File descriptor of the original stderr.

    Note:
        This function should be called after `redirect_all_output` to revert 
        to normal terminal output.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    os.close(stdout_fd)
    os.close(stderr_fd)



def train_yolo_model(folder_tree, **kwargs):
    """
    Trains a YOLOv8 model using the Ultralytics API and project-specific settings.

    This function:
    - Retrieves the YOLO model path and training parameters from HFinder_settings.
    - Builds the appropriate dataset and output paths.
    - Redirects all output (stdout and stderr) to a log file.
    - Trains the model using `YOLO.train()`.
    - Restores original output streams at the end.

    Parameters:
        folder_tree (dict): A dictionary representing the project folder hierarchy.
        **kwargs: Additional keyword arguments passed to `YOLO.train()`.
    """

    model = HFinder_settings.get("model")
    yolo = YOLO(model)
    data_yaml_path = HFinder_folders.get_subtree(folder_tree, "dataset/yaml")
    imgsz = HFinder_settings.get("target_size")[0]
    epochs = HFinder_settings.get("epochs")

    log_path = os.path.join(folder_tree["root"], "log/train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    HFinder_folders.append_subtree(folder_tree, "log", log_path)

    HFinder_log.info(f"Training YOLOv8 for {epochs} epochs with model {model}")
    stdout_fd, stderr_fd = redirect_all_output(log_path)
    try:       
        yolo.train(data=data_yaml_path,
                   project=os.path.join(folder_tree["root"], "runs"),
                   epochs=epochs,
                   imgsz=imgsz,
                   verbose=False,
                   **kwargs)
    finally:
        restore_output(stdout_fd, stderr_fd)

