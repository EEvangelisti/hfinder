"""
hfinder.core
============

This subpackage defines the *public API* of HFinder’s core utilities.

Exposed modules and functions:
- **log**: logging utilities (``set_verbosity``, ``info``, ``warn``, ``fail``)
- **utils**: helper functions for argument parsing, YAML loading, I/O redirection
- **geometry**: geometric helpers for bounding boxes and contours

Importing from ``hfinder.core`` gives direct access to these functions:

>>> from hfinder.core import info, sanitize, clamp_box_xyxy
"""

from .log import (
    set_verbosity,
    info,
    warn,
    fail
)

from .utils import (
    sanitize,
    load_argument_list,
    string_to_typefun,
    redirect_all_output,
    restore_output,
    save_yolo_segmentation_label,
    load_class_definitions_from_yaml,
    power_set
)

from .geometry import (
    is_valid_image_format,
    clamp_box_xyxy,
    contours_to_yolo_polygons,
    flat_to_pts_xy,
    bbox_xyxy_to_xywh
)

__all__ = [
    "set_verbosity",
    "info",
    "warn",
    "fail",
    "sanitize",
    "load_argument_list",
    "string_to_typefun",
    "redirect_all_output",
    "restore_output",
    "save_yolo_segmentation_label",
    "load_class_definitions_from_yaml",
    "power_set",
    "is_valid_image_format",
    "clamp_box_xyxy",
    "contours_to_yolo_polygons",
    "flat_to_pts_xy",
    "bbox_xyxy_to_xywh"
]
