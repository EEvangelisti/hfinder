"""
HFinder_segmentation

This module provides thresholding and segmentation utilities for multichannel 
images used in the HFinder pipeline. It supports both automatic and user-defined
thresholding strategies, as well as the integration of custom polygon 
annotations.

Functions in this module operate on single channels or full image stacks to 
produce binary masks and polygon representations, which can be used for 
downstream dataset generation or contour extraction.

Key features:
- Adaptive and fixed thresholding methods
- JSON-based custom segmentation support
- Modular design for compatibility with various image preprocessing workflows
"""

import os
import cv2
import json
import skimage
import numpy as np
import matplotlib.pyplot as plt
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings
import hfinder_geometry as HFinder_geometry



def auto_threshold_strategy(img, threshold):
    assert img.dtype == np.uint8, "Input image must be uint8"
    
    if False: # FIXME: Insert this with an option
        base = HFinder_settings.get("current_image.base")
        fig, _ = skimage.filters.try_all_threshold(img)
        root = HFinder_folders.get_masks_dir()
        output = os.path.join(root, f"{base}_all_threshold.jpg")
        print(output)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if threshold == "isodata":
        thresh = skimage.filters.threshold_isodata(img)
    elif threshold == "li":
        thresh = skimage.filters.threshold_li(img)
    elif threshold == "local":
        thresh = skimage.filters.threshold_local(img)
    elif threshold == "otsu":
        thresh = skimage.filters.threshold_otsu(img)
    elif threshold == "yen":
        thresh = skimage.filters.threshold_otsu(img)   
    elif threshold == "triangle":
        thresh = skimage.filters.threshold_triangle(img)   
    elif threshold == "auto":
        median_val = np.median(img)
        mean_val = np.mean(img)
        max_val = np.max(img)
        # Heuristic: minority signal often implies strong skew (median << mean)
        skew_ratio = (mean_val - median_val) / (max_val + 1e-5)
        if skew_ratio > 0.15:
            return auto_threshold_strategy(img, "triangle") 
        else:
            return auto_threshold_strategy(img, "otsu") 
    else:
        HFinder_log.fail(f"Unknown thresholding function '{threshold}'")

    binary = (img > thresh).astype(np.uint8) * 255
    return thresh, binary



def channel_custom_threshold(channel, threshold):
    """
    Apply custom thresholding to a single-channel image and extract YOLO-style 
    polygon annotations. This function performs binary thresholding followed by
    contour detection to identify regions of interest in an image channel. It 
    converts each contour to a flattened polygon in YOLO-normalized coordinates 
    (relative to the target image size).

    :param channel: 2D NumPy array representing a single grayscale image channel
    :type channel: np.ndarray
    :param threshold: threshold, which can be interpreted as:
        - a percentile if < 1 (e.g., 0.9 for the top 10% brightest pixels)
        - an absolute pixel intensity threshold (0–255) if ≥ 1
    :type threshold: float 
    :returns: A tuple comprising:
        - the binary thresholded image
        - a list of polygons where each polygon is a flattened list 
          [x1, y1, x2, y2, ..., xn, yn] in YOLO format (normalized to [0,1]).
    :retype: tuple[np.ndarray, List[List[float]]]
    """
    if isinstance(threshold, str):

        _, binary = auto_threshold_strategy(channel, threshold.lower())

    else:

        if threshold >= 1:
            thresh_val = threshold
        else:
            thresh_val = np.percentile(channel, threshold)

        _, binary = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    yolo_polygons = HFinder_geometry.contours_to_yolo_polygons(contours)

    return binary, yolo_polygons



def channel_auto_threshold(channel):
    """
    Apply automatic thresholding to a single-channel image to extract binary 
    masks and YOLO-style polygons. This is a wrapper around 
    `channel_custom_threshold`, using the default threshold value defined in 
    settings (in percent).

    :param channel: 2D NumPy array representing a single image channel
    :type channel: np.ndarray
    :returns:  A tuple comprising:
        - the binary thresholded image
        - a list of flattened polygons, each representing a detected object in 
          YOLO-normalized coordinates
    :retype: tuple[np.ndarray, List[List[float]]]
    """
    auto_threshold = HFinder_settings.get("auto_threshold")
    return channel_custom_threshold(channel, auto_threshold)



def channel_custom_segment(json_path, ratio):
    """
    Load and normalize segmentation polygons from a COCO-style JSON annotation 
    file. This function reads segmentation annotations from a JSON file and 
    rescales the coordinates to match the normalized YOLO format, according to 
    a given `ratio` and the target image size defined in settings.

    :param json_path: Path to the JSON file containing COCO-style annotations.
    :type json_path: str
    :param ratio: Scaling ratio between original image dimensions and the 
                  target size. Used to rescale polygon coordinates.
    :type ratio: float
    :returns: a list of flattened polygons, where each polygon is represented 
        as a list of alternating x and y coordinates (normalized to [0,1]).
    :retype: List[List[float]]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    w, h = HFinder_settings.get("target_size")

    polygons = []
    for ann in data.get("annotations", []):
        if "segmentation" not in ann or not ann["segmentation"]:
            continue

        for seg in ann["segmentation"]:
            if not seg or len(seg) % 2 != 0:
                HFinder_log.warn(f"Invalid segmentation in {json_path}, " + \
                                 f"annotation id {ann.get('id')}")
                continue

            flat = [seg[i] * ratio / w if i % 2 == 0 else seg[i] * ratio / h 
                    for i in range(len(seg))]
            polygons.append(flat)

    return polygons



