# HFinder

"""
hfinder_segmentation.py

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

import cv2
import json
import numpy as np
import hfinder_log as HFinder_log
import hfinder_settings as HFinder_settings
import hfinder_geometry as HFinder_geometry



def auto_threshold_strategy(img, threshold):
    """
    Automatically selects the best thresholding method (OTSU or Triangle) 
    depending on whether the signal is minority or not.
    
    Parameters:
        img (np.ndarray): Grayscale image (uint8).
        threshold (str): thresholding function (auto, otsu, triangle).
    
    Returns:
        binary_mask (np.ndarray): Binary image after thresholding.
    """
    assert img.dtype == np.uint8, "Input image must be uint8"
    
    if threshold == "otsu":
        flag = cv2.THRESH_OTSU
    elif threshold == "triangle":
        flag = cv2.THRESH_TRIANGLE
    elif threshold == "auto":
        median_val = np.median(img)
        mean_val = np.mean(img)
        max_val = np.max(img)
        # Heuristic: minority signal often implies strong skew (median << mean)
        skew_ratio = (mean_val - median_val) / (max_val + 1e-5)
        flag = cv2.THRESH_TRIANGLE if skew_ratio > 0.15 else cv2.THRESH_OTSU
    else:
        HFinder_log.fail(f"Unknown thresholding function '{threshold}'")
    
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + flag)



def channel_custom_threshold(channel, threshold):
    """
    Apply custom thresholding to a single-channel image and extract YOLO-style 
    polygon annotations.

    This function performs binary thresholding followed by contour detection to 
    identify regions of interest in an image channel. It converts each contour 
    to a flattened polygon in YOLO-normalized coordinates (relative to the 
    target image size).

    Args:
        channel (np.ndarray): 2D NumPy array representing a single grayscale image channel.
        threshold (float): 
            - If < 1, interpreted as a percentile (e.g., 0.9 for the top 10% brightest pixels).
            - If ≥ 1, interpreted as an absolute pixel intensity threshold (0–255).

    Returns:
        tuple:
            - binary (np.ndarray): Binary thresholded image.
            - yolo_polygons (List[List[float]]): List of polygons where each polygon is a flattened list 
              [x1, y1, x2, y2, ..., xn, yn] in YOLO format (normalized to [0,1]).

    Notes:
        - The image size is taken from `HFinder_settings.get("target_size")`.
        - Only external contours are retained.
        - Contours with fewer than 3 points are discarded.
    """
    if isinstance(threshold, str):
        _, binary = auto_threshold_strategy(channel, threshold.lower())
    else:
        thresh_val = threshold if threshold >= 1 else np.percentile(channel, threshold)
        _, binary = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY)

    w, h = HFinder_settings.get("target_size")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_polygons = HFinder_geometry.contours_to_yolo_polygons(contours)
    return binary, yolo_polygons



def channel_auto_threshold(channel):
    """
    Apply automatic thresholding to a single-channel image to extract binary 
    masks and YOLO-style polygons.

    This is a wrapper around `channel_custom_threshold`, using the default threshold
    value defined in settings (in percent).

    Args:
        channel (np.ndarray): 2D NumPy array representing a single image channel.

    Returns:
        tuple:
            - binary (np.ndarray): Binary image resulting from thresholding.
            - yolo_polygons (List[List[float]]): List of flattened polygons, each representing
              a detected object in YOLO-normalized coordinates.
    """
    auto_threshold = HFinder_settings.get("default_auto_threshold")
    return channel_custom_threshold(channel, auto_threshold)



def channel_custom_segment(json_path, ratio):
    """
    Load and normalize segmentation polygons from a COCO-style JSON annotation file.

    This function reads segmentation annotations from a JSON file and rescales 
    the coordinates to match the normalized YOLO format, according to a given 
    `ratio` and the target image size defined in settings.

    Args:
        json_path (str): Path to the JSON file containing COCO-style annotations.
        ratio (float): Scaling ratio between original image dimensions and the 
                       target size. Used to rescale polygon coordinates.

    Returns:
        List[List[float]]: A list of flattened polygons, where each polygon is represented 
        as a list of alternating x and y coordinates (normalized to [0,1]).

    Notes:
        - Only valid segmentation entries are processed.
        - Segments with missing or malformed data (e.g., odd number of coordinates) are skipped.
        - Coordinate normalization assumes the original segmentation is in absolute pixels
          and applies:  
              `x_new = x * ratio / width`,  
              `y_new = y * ratio / height`.
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
                HFinder_log.warn(f"Invalid segmentation in {json_path}, annotation id {ann.get('id')}")
                continue

            flat = [seg[i] * ratio / w if i % 2 == 0 else seg[i] * ratio / h for i in range(len(seg))]
            polygons.append(flat)

    return polygons
