import os
import sys
import yaml
import numpy as np
from itertools import combinations
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



def write_yolo_yaml(class_ids):
    """
    Generate and save a YOLO-compatible dataset YAML file. This function creates
    a `dataset.yaml` file describing the training and validation dataset 
    structure for YOLOv8, including class names, number of classes, and paths
    to training/validation images. The file is written to `dataset/dataset.yaml`
    within the project's root directory.

    Parameters:
        class_ids (dict): A dictionary mapping class names (str) to class 
                          indices (int). Example: {"cell": 0, "noise": 1}
    """
    data = {
        "path": HFinder_folders.get_root(),
        "train": HFinder_folders.get_image_train_dir(),
        "val": HFinder_folders.get_image_val_dir(),
        "nc": len(class_ids),
        "names": [x for x, _ in sorted(class_ids.items(), key=lambda x: x[1])]
    }

    yaml_path = os.path.join(HFinder_folders.get_dataset_dir(), "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)



def save_yolo_segmentation_label(file_path, polygons, class_ids):
    """
    Writes YOLOv8-style segmentation labels to a .txt file, with one line per 
    polygon, using normalized coordinates.

    Parameters:
        file_path (str): Path to the output label file.
        polygons (list[tuple[str, list[tuple[float, float]]]]): List of tuples 
        (class_name, polygon).
        class_ids (dict[str, int]): Mapping from class names to YOLO integer IDs.
    """
    with open(file_path, "w") as f:
        for class_name, poly in polygons:
            if class_name not in class_ids:
                continue
            class_id = class_ids[class_name]
            poly = [coord for point in poly for coord in point]
            line = [str(class_id)] + [f"{x:.6f}" for x in poly]
            f.write(" ".join(line) + "\n")



def power_set(channels, n, c):
    if n == 1:
        s = list(channels)
        return [combo for r in range(1, 4) for combo in combinations(s, r)]
    
    all_combos = []
    for stack_index in range(n):
        # Canaux de ce stack présents dans `channels`
        group = [ch for ch in channels if (ch - 1) // c == stack_index]
        for r in range(1, 4):
            all_combos.extend(combinations(group, r))
    return all_combos


