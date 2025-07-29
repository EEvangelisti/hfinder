import os
import sys
import yaml
from ultralytics import YOLO

import hfinder_folders as HFinder_folders



def write_yolo_dataset_yaml(folder_tree, class_names, output_name="dataset.yaml"):
    """
    Create a YAML file describing the dataset for YOLOv8 training and add it to the folder tree.

    Args:
        folder_tree (dict): The tree structure containing dataset paths.
        class_names (list of str): List of class names (e.g., ["hyphae"]).
        output_name (str): Name of the YAML file to generate (default: "dataset.yaml").

    Returns:
        str: Full path to the created YAML file.
    """

    dataset_root = folder_tree["root"]
    train_images = os.path.join(dataset_root, "dataset/images/train")
    val_images = os.path.join(dataset_root, "dataset/images/val")

    data_yaml = {
        "path": dataset_root,
        "train": "dataset/images/train",
        "val": "dataset/images/val",
        "names": {i: name for i, name in enumerate(class_names)}
    }

    yaml_path = os.path.join(dataset_root, output_name)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    # Update folder tree
    HFinder_folders.set_subtree(folder_tree, "dataset/yaml", yaml_path)

    return yaml_path



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



def train_yolo_model_hyphae(folder_tree, model="yolov8n.pt", epochs=100, imgsz=640, **kwargs):
    """
    Train a YOLOv8 model using Ultralytics CLI.

    Args:
        data_yaml_path (str): Path to the dataset YAML file.
        model (str): Base model to fine-tune (e.g., "yolov8n.pt", "yolov8s.pt").
        epochs (int): Number of training epochs.
        imgsz (int): Input image size (default: 640).
    """
    data_yaml_path = write_yolo_dataset_yaml(folder_tree, ["hyphae"])
    yolo = YOLO(model)

    log_path = os.path.join(folder_tree["root"], "log/train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    HFinder_folders.append_subtree(folder_tree, "log", log_path)

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








