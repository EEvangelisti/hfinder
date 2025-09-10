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
- All paths resolve via hf_folders and hf_settings to keep I/O centralized.
"""

import os
import sys
import json
import yaml
import numpy as np
import importlib.resources as ir
from itertools import combinations
from hfinder.core import hf_log as HF_log
from hfinder.core import hf_folders as HF_folders



def load_argument_list(filename):
    """
    Load a JSON argument list bundled in the ``hfinder.data`` package.

    :param filename: Name of the JSON resource file (e.g. ``"annot2images.arglist.json"``).
    :type filename: str
    :return: Parsed JSON content as a Python object (usually a dict or list).
    :rtype: dict or list
    :raises FileNotFoundError: If the specified file does not exist in the package.
    :raises json.JSONDecodeError: If the file content is not valid JSON.
    """
    path = ir.files("hfinder.data").joinpath(filename)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def string_to_typefun(type_str):
    """
    Convert a string to a Python type function usable in argparse.

    :param type_str: Name of the type as a string (e.g. "int", "float", "str").
    :type type_str: str
    :return: The corresponding Python callable (e.g. int, float, str),
             or the input unchanged if not recognized.
    :rtype: callable
    """
    TYPE_MAP = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool
    }
    return TYPE_MAP.get(type_str, type_str)



def redirect_all_output(log_path):
    """
    Redirect both stdout and stderr to a specified log file.

    Works cross-platform (Unix/Windows) by reassigning sys.stdout/sys.stderr.

    :param log_path: Path to the log file where output will be written.
    :type log_path: str
    :return: Tuple of the original stdout and stderr objects.
    :rtype: tuple[TextIO, TextIO]
    """
    # Flush existing buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Open log file in append mode so output isn't lost on multiple calls
    log_file = open(log_path, "w", buffering=1, encoding="utf-8")

    # Save original stdout and stderr
    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    # Redirect to log file
    sys.stdout = log_file
    sys.stderr = log_file

    return orig_stdout, orig_stderr



def restore_output(orig_stdout, orig_stderr):
    """
    Restore stdout and stderr to their original state.

    :param orig_stdout: The original sys.stdout.
    :type orig_stdout: TextIO
    :param orig_stderr: The original sys.stderr.
    :type orig_stderr: TextIO
    """
    sys.stdout.flush()
    sys.stderr.flush()

    # Restore
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr



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
        "path": HF_folders.get_root(),
        "train": HF_folders.get_image_train_dir(),
        "val": HF_folders.get_image_val_dir(),
        "nc": len(class_ids),
        "names": [x for x, _ in sorted(class_ids.items(), key=lambda x: x[1])]
    }

    yaml_path = os.path.join(HF_folders.get_dataset_dir(), "dataset.yaml")
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



def load_class_definitions_from_yaml(yaml_path):
    """
    Read YOLO dataset.yaml and return a mapping {class_name: class_id}.
    Supports both:
      - names: [ "class0", "class1", ... ]
      - names: {0: "class0", 1: "class1", ...}
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if not names:
        raise ValueError(f"No 'names' in {yaml_path}")

    if isinstance(names, list):
        return {name: i for i, name in enumerate(names)}
    elif isinstance(names, dict):
        # keys can be str or int -> ensure int ids
        return {name: int(i) for i, name in names.items()}
    else:
        raise TypeError(f"'names' must be list or dict in {yaml_path}, got {type(names).__name__}")


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


