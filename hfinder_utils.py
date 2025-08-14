"""
General utilities for HFinder.

This module centralizes:
- Output redirection to a session log (stdout/stderr) and safe restoration.
- YOLO dataset YAML generation from class definitions.
- Writing YOLOv8 segmentation labels (polygon format).
- Lightweight helpers for discovering class names and building channel subsets.

Public API
----------
- redirect_all_output(log_path): Redirect stdout/stderr to a log file.
- restore_output(stdout_fd, stderr_fd): Restore the original output streams.
- write_yolo_yaml(class_ids): Emit a YOLO dataset YAML in the session dataset dir.
- save_yolo_segmentation_label(file_path, annotations, class_ids): Write YOLO seg labels.
- load_class_definitions(): Discover classes from JSON files under tiff_dir/classes.
- power_set(channels, n, c): Build small channel combinations per z-stack.

Notes
-----
- All paths resolve via hfinder_folders and hfinder_settings to keep I/O centralized.
"""

import os
import sys
import yaml
import numpy as np
from glob import glob
from itertools import combinations
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



def redirect_all_output(log_path):
    """
    Redirect both stdout and stderr to a specified log file.

    This function duplicates the current stdout/stderr file descriptors,
    redirects them to *log_path*, and returns the originals so they can be
    restored later with restore_output().

    :param log_path: Path to the log file where output will be written.
    :type log_path: str
    :return: File descriptors for the original stdout and stderr.
    :rtype: tuple[int, int]
    """
    
    # Flush any buffered output before switching streams
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Open/truncate the log file; get a writable file descriptor (FD)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    
    # Save original stdout and stderr FDs
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)

    # Redirect stdout and stderr to the log file FD
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    return stdout_fd, stderr_fd



def restore_output(stdout_fd, stderr_fd):
    """
    Restore stdout and stderr to their original state.

    Call this after redirect_all_output() to revert logging back to the terminal.

    :param stdout_fd: File descriptor of the original stdout.
    :type stdout_fd: int
    :param stderr_fd: File descriptor of the original stderr.
    :type stderr_fd: int
    :rtype: None
    """
    
    # Flush any pending data in the redirected streams
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Rebind stdout/stderr to their original FDs and close the temporaries
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    os.close(stdout_fd)
    os.close(stderr_fd)



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
        "path": HFinder_folders.get_root(),
        "train": HFinder_folders.get_image_train_dir(),
        "val": HFinder_folders.get_image_val_dir(),
        "nc": len(class_ids),
        "names": [x for x, _ in sorted(class_ids.items(), key=lambda x: x[1])]
    }

    yaml_path = os.path.join(HFinder_folders.get_dataset_dir(), "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)



def save_yolo_segmentation_label(file_path, annotations, class_ids):
    """
    Write YOLOv8 segmentation labels (polygon format).

    Input annotations are already normalized to [0, 1] and provided as flat
    polygons per object.

    :param file_path: Output label path (typically under dataset/labels/*/*.txt).
    :type file_path: str
    :param annotations: Iterable of (cls_name, polygons) where
                        polygons = [[x1, y1, x2, y2, ...], ...] (normalized).
    :type annotations: list[tuple[str, list[list[float]]]]
    :param class_ids: Mapping from class name to class index.
    :type class_ids: dict[str, int]
    :rtype: None
    """
    with open(file_path, "w") as f:
        for cls_name, polygons in annotations:
            cls_id = class_ids.get(cls_name)
            if cls_id is None:
                continue
            for flat in polygons:  # UNE LIGNE PAR POLYGONE
                f.write(f"{cls_id} " + " ".join(f"{v:.6f}" for v in flat) + "\n")



def load_class_definitions():
    """
    Extract class names from JSON files in `tiff_dir/classes/` and assign IDs.

    Class names are inferred from file basenames, sorted alphabetically, and
    assigned increasing integer IDs starting at 0.

    :return: Mapping from class name to class ID, e.g., {"hyphae": 0, "nuclei": 1}.
    :rtype: dict[str, int]
    """
    class_dir = os.path.join(HFinder_settings.get("tiff_dir"), "classes")
    if not os.path.isdir(class_dir):
        HFinder_log.fail(f"No such directory: {class_dir}")

    files = sorted(glob(os.path.join(class_dir, "*.json")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return {name: i for i, name in enumerate(names)}



def power_set(channels, n, c):
    """
    Build small combinations of channels for visualization or modeling.

    If `n == 1` (single plane), return all 1-, 2-, and 3-channel combinations
    from `channels`. If `n > 1` (z-stack present), combinations are generated
    **per stack index** (i.e., channels that belong to the same z-slice), and
    all such combinations are returned.

    Channel indexing convention:
      - Channels are numbered 1..(n*c).
      - For a given channel `ch`, the zero-based stack index is `(ch - 1) // c`.

    :param channels: Iterable of channel indices to consider (1-based).
    :type channels: Iterable[int]
    :param n: Number of z-slices (1 for single plane).
    :type n: int
    :param c: Number of channels per slice.
    :type c: int
    :return: List of tuples representing channel combinations.
    :rtype: list[tuple[int, ...]]
    """
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


