import os
from datetime import datetime
import hfinder_log as HFinder_log

SESSION_TREE = {}

def get_timestamp():
    """
    Returns the current date and time as a string formatted as 'YYYY-MM-DD_HH-MM-SS'.
    This format is suitable for creating unique, chronologically ordered folder or file names.
    Returns:
        str: The current timestamp, e.g., '2025-07-27_18-24-03'.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_tree(base, tree):
    """
    Recursively creates a directory structure from a nested dictionary.
    Each key in the dictionary represents a folder name. If the value is another 
    dictionary, the function recurses into it. If the value is not a dictionary 
    (e.g., a list, None, etc.), the folder is created but not recursed further.

    Args:
        base (str): The root path where the directory tree should be created.
        tree (dict): A nested dictionary representing the folder hierarchy.
    """
    os.makedirs(base, exist_ok=True)
    if isinstance(tree, dict):
        for name, subtree in tree.items():
            path = os.path.join(base, name)
            create_tree(path, subtree)       


def create_session_folders():
    """
    Creates a timestamped root directory containing a standard dataset structure 
    for machine learning training and validation, and returns the full tree.

    The structure is as follows:
        <timestamp>/
        ├── dataset/
        │   ├── masks/
        │   ├── contours/
        │   ├── images/
        │   │   ├── train/
        │   │   └── val/
        │   └── labels/
        │       ├── train/
        │       └── val/
        ├── log/
        └── runs/

    Returns:
        dict: A nested dictionary representing the folder structure with an additional
              "root" key pointing to the absolute path of the created base directory.
    """
    base = get_timestamp()
    HFinder_log.info(f"Creating session folder {base}")
    folder_tree = {
        "dataset": {
            "masks": [],
            "contours": [],
            "images": {
                "train": [],
                "val": [],
            },
            "labels": {
                "train": [],
                "val": [],
            },
        },
        "runs": [],
        "log": []
    }
    create_tree(base, folder_tree)
    folder_tree["root"] = os.path.abspath(base)
    global SESSION_TREE
    SESSION_TREE = folder_tree

def get_root():
    return SESSION_TREE["root"]

def rootify(path):
    return os.path.join(SESSION_TREE["root"], path)

def get_log_dir(root=True):
    path = "log"
    return rootify(path) if root else path

def get_runs_dir(root=True):
    path = "runs"
    return rootify(path) if root else path

def get_dataset_dir(root=True):
    path = "dataset"
    return rootify(path) if root else path

def get_image_train_dir(root=True):
    path = os.path.join("dataset", "images", "train")
    return rootify(path) if root else path

def get_label_train_dir(root=True):
    path = os.path.join("dataset", "labels", "train")
    return rootify(path) if root else path

def get_image_val_dir(root=True):
    path = os.path.join("dataset", "images", "val")
    return rootify(path) if root else path

def get_label_val_dir(root=True):
    path = os.path.join("dataset", "labels", "val")
    return rootify(path) if root else path

def get_masks_dir(root=True):
    path = os.path.join("dataset", "masks")
    return rootify(path) if root else path

def get_contours_dir(root=True):
    path = os.path.join("dataset", "contours")
    return rootify(path) if root else path


