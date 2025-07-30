import os
import sys
import yaml
from ultralytics import YOLO
import hfinder_log as HFinder_log 
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



def redirect_all_output(log_path):
    """
    Redirects all output (stdout and stderr) to a log file using low-level file descriptors.
    Returns a tuple of (original_stdout_fd, original_stderr_fd) for later restoration.
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
    Restores the original stdout and stderr using saved file descriptors.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    os.close(stdout_fd)
    os.close(stderr_fd)



def train_yolo_model(folder_tree, **kwargs):
    """
    Train a YOLOv8 model using Ultralytics CLI.

    Args:
        data_yaml_path (str): Path to the dataset YAML file.
        model (str): Base model to fine-tune (e.g., "yolov8n.pt", "yolov8s.pt").
        epochs (int): Number of training epochs.
        imgsz (int): Input image size (default: 640).
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








