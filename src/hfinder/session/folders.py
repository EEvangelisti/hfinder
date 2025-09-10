"""
Session folder utilities for HFinder.

This module creates and manages a timestamped "session" directory tree used
to organize datasets, logs, and run artifacts in a consistent layout.

Overview
--------
- A new session is created under a root folder named with the current timestamp:
      YYYY-MM-DD_HH-MM-SS/
  This deterministic, lexicographically sortable format eases archival and lookup.

- The standard structure is:
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

Public API
----------
- create_session_folders(): Create the tree and record its absolute root.
- get_root(): Return the absolute path to the current session root.
- rootify(path): Make a path absolute by prefixing the session root.
- Directory helpers (optionally absolute with root=True):
  - get_log_dir(root=True)
  - get_runs_dir(root=True)
  - get_dataset_dir(root=True)
  - get_image_train_dir(root=True)
  - get_label_train_dir(root=True)
  - get_image_val_dir(root=True)
  - get_label_val_dir(root=True)
  - get_masks_dir(root=True)
  - get_contours_dir(root=True)

Notes
-----
- The global SESSION_TREE is populated by create_session_folders() and contains
  the nested structure plus a "root" key with the absolute path.
"""

import os
import yaml
from datetime import datetime
from hfinder.core import hf_log as HF_log


# In-memory description of the session folder tree.
# Populated by create_session_folders(); includes a "root" absolute path.
SESSION_TREE = {}


def sanity_check():
    """
    Ensure that the session folder structure has been initialized.

    This function asserts that the global SESSION_TREE is non-empty,
    which indicates that create_session_folders() has been called
    successfully. It is intended as a development-time safeguard:
    if this assertion fails, the program flow is incorrect and the
    calling code must be fixed.

    :raises AssertionError: If SESSION_TREE is empty (session not initialized).
    :rtype: None
    """
    assert SESSION_TREE, (
        "(HFinder) Assert Failure: Session not initialized. "
        "Call create_session_folders() first."
    )


def get_timestamp():
    """
    Format the current local date/time as a sortable session name.

    The pattern is 'YYYY-MM-DD_HH-MM-SS', which sorts chronologically and
    is safe for use as a directory name on common filesystems.

    :return: Timestamp string, e.g., '2025-07-27_18-24-03'.
    :rtype: str
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_tree(base, tree):
    """
    Recursively create a directory structure from a nested mapping.

    Each key in *tree* is a folder name under *base*. If a value is a dict,
    recursion continues into that sub-tree. Non-dict values (e.g., list/None)
    indicate leaf folders that should be created but not descended into.

    :param base: Root path where the directory tree should be created.
    :type base: str
    :param tree: Nested mapping representing folders and subfolders.
    :type tree: dict
    :rtype: None
    """
    
    # Ensure the current base exists.
    os.makedirs(base, exist_ok=True)
    
    # Only dicts imply nested subfolders; anything else is treated as a leaf.
    if isinstance(tree, dict):
        for name, subtree in tree.items():
            path = os.path.join(base, name)
            create_tree(path, subtree)       


def create_session_folders():
    """
    Create a fresh, timestamped session directory with the standard layout.

    On completion, the global SESSION_TREE is populated with the nested folder
    structure and a "root" entry pointing to the absolute session root.

    :return: None
    :rtype: None
    """
    base = get_timestamp()
    HF_log.info(f"Creating session folder {base}")
    
    # Declarative specification of the expected subfolders.
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
    
    # Materialize the tree on disk.
    create_tree(base, folder_tree)
    
    # Record the absolute session root and publish the tree globally.
    folder_tree["root"] = os.path.abspath(base)
    global SESSION_TREE
    SESSION_TREE = folder_tree


def get_root():
    """
    Get the absolute path to the current session root.

    :return: Absolute session root directory.
    :rtype: str
    """
    sanity_check()
    return SESSION_TREE["root"]


def rootify(path):
    """
    Prefix a relative path with the current session root.

    :param path: Relative path inside the session.
    :type path: str
    :return: Absolute path rooted at the session directory.
    :rtype: str
    """
    sanity_check()
    return os.path.join(SESSION_TREE["root"], path)


def get_log_dir(root=True):
    """
    Get the path to the 'log' directory.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'log'.
    :rtype: str
    """
    path = "log"
    sanity_check()
    return rootify(path) if root else path


def get_runs_dir(root=True):
    """
    Get the path to the 'runs' directory.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'runs'.
    :rtype: str
    """
    path = "runs"
    sanity_check()
    return rootify(path) if root else path


def get_dataset_dir(root=True):
    """
    Get the path to the 'dataset' directory.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'dataset'.
    :rtype: str
    """
    path = "dataset"
    sanity_check()
    return rootify(path) if root else path


def get_image_train_dir(root=True):
    """
    Get the path to 'dataset/images/train'.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'dataset/images/train'.
    :rtype: str
    """
    sanity_check()
    path = os.path.join("dataset", "images", "train")
    return rootify(path) if root else path


def get_label_train_dir(root=True):
    """
    Get the path to 'dataset/labels/train'.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'dataset/labels/train'.
    :rtype: str
    """
    sanity_check()
    path = os.path.join("dataset", "labels", "train")
    return rootify(path) if root else path


def get_image_val_dir(root=True):
    """
    Get the path to 'dataset/images/val'.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'dataset/images/val'.
    :rtype: str
    """
    sanity_check()
    path = os.path.join("dataset", "images", "val")
    return rootify(path) if root else path


def get_label_val_dir(root=True):
    """
    Get the path to 'dataset/labels/val'.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'dataset/labels/val'.
    :rtype: str
    """
    sanity_check()
    path = os.path.join("dataset", "labels", "val")
    return rootify(path) if root else path


def get_masks_dir(root=True):
    """
    Get the path to 'dataset/masks'.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'dataset/masks'.
    :rtype: str
    """
    sanity_check()
    path = os.path.join("dataset", "masks")
    return rootify(path) if root else path


def get_contours_dir(root=True):
    """
    Get the path to 'dataset/contours'.

    :param root: If True, return an absolute path; otherwise relative.
    :type root: bool
    :return: Path to 'dataset/contours'.
    :rtype: str
    """
    sanity_check()
    path = os.path.join("dataset", "contours")
    return rootify(path) if root else path



def write_yolo_yaml(class_ids):
    """
    Generate and save a YOLO-compatible dataset YAML file.

    The file `dataset.yaml` is written under the session dataset directory and
    contains:
      - `path`: absolute project root,
      - `train`/`val`: absolute paths to image folders,
      - `nc`: number of classes,
      - `names`: class names ordered by class index.

    :param class_ids: Mapping from class name to class index
                      (e.g., {"cell": 0, "noise": 1}).
    :type class_ids: dict[str, int]
    :rtype: None
    """
    data = {
        "path": get_root(),
        "train": get_image_train_dir(),
        "val": get_image_val_dir(),
        "nc": len(class_ids),
        "names": [x for x, _ in sorted(class_ids.items(), key=lambda x: x[1])]
    }

    yaml_path = os.path.join(get_dataset_dir(), "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)



