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
from skimage.morphology import binary_closing
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings
import hfinder_geometry as HFinder_geometry


def is_bool(image):
    """
    Return True if the array dtype is boolean.

    :param image: Input array.
    :type image: np.ndarray
    :returns: True if dtype is np.bool_ (or bool), False otherwise.
    :rtype: bool
    """
    return image.dtype == np.bool_ or image.dtype == bool



def to_bool(image):
    """
    Convert an array to a boolean mask.

    Rules:
        - If the array is already boolean, it is returned as-is.
        - Otherwise, strictly positive values are mapped to True (foreground),
          and zeros/negatives to False (background).

    :param image: Input array (bool, uint8 0/255, float, etc.).
    :type image: np.ndarray
    :returns: Boolean mask with the same shape as the input.
    :rtype: np.ndarray
    """
    return image if is_bool(image) else image > 0



def to_uint8(image):
    """
    Convert an array to uint8 (0–255) using ``skimage.util.img_as_ubyte``.

    Notes:
        - bool → {False, True} maps to {0, 255}
        - float in [0, 1] → rescaled to [0, 255]
        - other numeric dtypes are handled per scikit-image conversion rules
          (values may be scaled/clipped accordingly)

    :param image: Input array.
    :type image: np.ndarray
    :returns: uint8 array in the range 0–255.
    :rtype: np.ndarray
    """
    return skimage.util.img_as_ubyte(image)



def fill_gaps(binary, area_threshold=50, radius=1, connectivity=2):
    """
    Fill small black holes inside white foreground and then perform a
    morphological closing.

    Pipeline:
        1) ``remove_small_holes`` fills cavities with area ≤ ``area_threshold``.
        2) ``binary_closing`` with a disk structuring element (size parameter
           ``radius`` is passed to ``skimage.morphology.disk``).

    :param binary: Boolean mask (True = foreground).
    :type binary: np.ndarray
    :param area_threshold: Maximum hole area (in pixels) to be filled.
    :type area_threshold: int
    :param radius: Size parameter forwarded as the *radius* to
                     ``skimage.morphology.disk``.
    :type diameter: int
    :param connectivity: Pixel connectivity (2 → 8-connectivity in 2D).
    :type connectivity: int
    :returns: Boolean mask with small holes filled and small gaps closed.
    :rtype: np.ndarray
    :raises AssertionError: If ``binary`` is not a boolean mask.
    """
    assert is_bool(binary), "(HFinder) Assert Failure: fill_gaps"
    filled_bool = remove_small_holes(binary,
                                     area_threshold=area_threshold,
                                     connectivity=connectivity)
    disk_1 = skimage.morphology.disk(radius)
    closed_bool = binary_closing(filled_bool, footprint=disk_1)
    return closed_bool



# TODO: Alternative methods we should consider in the future:
# For example, we could define an option to choose which noise removal
# function to use.
# disk1 = skimage.morphology.disk(1)
# clean = skimage.morphology.opening(mask_bool, footprint=disk1)
# clean = skimage.filters.median(mask_bool, footprint=disk1).astype(bool)
def remove_noise(binary, min_size=20, connectivity=2):
    """
    Remove small white connected components (“speckles”) by minimum area.

    :param binary: Boolean mask (True = foreground).
    :type binary: np.ndarray
    :param min_size: Minimum area (in pixels) for components to keep.
    :type min_size: int
    :param connectivity: Pixel connectivity (2 → 8-connectivity in 2D).
    :type connectivity: int
    :returns: Boolean mask with small components removed.
    :rtype: np.ndarray
    :raises AssertionError: If ``binary`` is not a boolean mask.
    """
    assert is_bool(binary), "(HFinder) Assert Failure: remove_noise"
    return remove_small_objects(binary,
                                min_size=min_size,
                                connectivity=connectivity)



def noise_and_gaps(img):
    """
    Convenience post-processing chain that returns a clean 0/255 mask.

    Steps:
        - Convert to boolean via ``to_bool``.
        - Remove small components via ``remove_noise``.
        - Fill small holes + closing via ``fill_gaps``.
        - Convert back to uint8 0/255 via ``to_uint8``.

    :param img: Input mask or image (bool, 0/255, float, etc.).
    :type img: np.ndarray
    :returns: Cleaned binary mask (uint8, values in {0, 255}).
    :rtype: np.ndarray
    """
    return to_uint8(fill_gaps(remove_noise(to_bool(img))))



def auto_threshold_strategy(img, threshold):
    """
    Apply a named thresholding method (scikit-image) and return the cleaned mask.

    Supported methods:
        - "isodata", "li", "otsu", "yen", "triangle"
        - "auto": heuristic based on skewness:
          if (mean - median) / (max + 1e-5) > 0.15 → "triangle", else "otsu".

    Post-processing:
        The binary mask (``img > thresh``) is cleaned via ``noise_and_gaps``
        before being returned.

    :param img: Grayscale image (must be uint8).
    :type img: np.ndarray
    :param threshold: Method name (see above).
    :type threshold: str
    :returns: (numeric threshold, cleaned binary mask as uint8 0/255).
    :rtype: tuple[float, np.ndarray]
    :raises AssertionError: If ``img`` is not uint8.
    """
    assert img.dtype == np.uint8, "(HFinder) Assert Failure: auto_threshold_strategy"
    
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
    elif threshold == "otsu":
        thresh = skimage.filters.threshold_otsu(img)
    elif threshold == "yen":
        thresh = skimage.filters.threshold_yen(img)   
    elif threshold == "triangle":
        thresh = skimage.filters.threshold_triangle(img)   
    elif threshold == "auto":
        median_val = np.median(img)
        mean_val = np.mean(img)
        max_val = np.max(img)
        # Heuristic: minority signal often implies strong skew (median << mean)
        skew_ratio = (mean_val - median_val) / (max_val + 1e-5)
        return auto_threshold_strategy(img,"triangle" if skew_ratio > 0.15 else "otsu")
    else:
        HFinder_log.warn(f"Unknown thresholding function '{threshold}'")
        return auto_threshold_strategy(img, "otsu")

    return float(thresh), noise_and_gaps(img > thresh)



def channel_custom_threshold(channel, threshold):
    """
    Threshold a single channel and extract YOLO-style polygons.

    Behavior:
        - If ``threshold`` is a string, use ``auto_threshold_strategy`` and
          post-process via ``noise_and_gaps``.
        - If ``threshold`` is numeric:
            * ``threshold`` < 1 → percentile (e.g., 0.9 = 90th percentile)
            * ``threshold`` ≥ 1 → absolute intensity threshold (0–255)
          Then threshold via OpenCV and post-process via ``noise_and_gaps``.

    Contours are extracted with ``cv2.findContours`` and converted to
    flattened polygons in YOLO-normalized coordinates.

    :param channel: 2D single-channel image (uint8 recommended).
    :type channel: np.ndarray
    :param threshold: Method name, percentile fraction, or absolute threshold.
    :type threshold: str | float
    :returns: (cleaned binary mask, list of flattened YOLO-normalized polygons).
    :rtype: tuple[np.ndarray, list[list[float]]]
    """
    if isinstance(threshold, str):

        _, binary = auto_threshold_strategy(channel, threshold.lower())

    else:

        if threshold < 1:
            thresh_val = np.percentile(channel, threshold * 100)
        else:
            thresh_val = threshold

        _, binary = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY)
        binary = noise_and_gaps(binary)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    yolo_polygons = HFinder_geometry.contours_to_yolo_polygons(contours)

    return binary, yolo_polygons



def channel_auto_threshold(channel):
    """
    Wrapper around ``channel_custom_threshold`` that uses the default threshold
    from settings (``HFinder_settings.get("auto_threshold")``).

    :param channel: 2D single-channel image (uint8 recommended).
    :type channel: np.ndarray
    :returns: (cleaned binary mask, list of flattened YOLO-normalized polygons).
    :rtype: tuple[np.ndarray, list[list[float]]]
    """
    auto_threshold = HFinder_settings.get("auto_threshold")
    return channel_custom_threshold(channel, auto_threshold)



def channel_custom_segment(json_path, ratio):
    """
    Load COCO-style segmentation polygons and normalize them to YOLO format.

    Processing:
        - Read ``annotations[*].segmentation`` from the JSON file.
        - Get the target size (``w, h``) from settings.
        - Rescale coordinates by ``ratio`` and normalize to [0, 1] by dividing
          x by ``w`` and y by ``h``.
        - Skip empty/odd-length segmentations (a warning is logged).

    :param json_path: Path to the COCO-style JSON file.
    :type json_path: str
    :param ratio: Scale factor between the original image and the target size.
    :type ratio: float
    :returns: List of flattened polygons [x1, y1, ..., xn, yn] normalized to [0, 1].
    :rtype: list[list[float]]
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



