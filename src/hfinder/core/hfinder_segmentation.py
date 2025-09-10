"""
Label-generation helpers for confocal images (dark background, grayscale).

Overview
--------
- Enforce 8-bit discipline (0–255) with explicit conversion helpers.
- Threshold (named methods or numeric) → clean up → (optional) watershed split.
- Extract external contours and (optionally) simplify them.
- Convert masks/contours into YOLO-normalized polygons.
- Load COCO-style polygons and normalize to YOLO format.

Public API
----------
- mask_to_polygons(mask, mode="subtract_holes", ...): Binary mask → polygons (with/without holes).
- is_bool(image): Check if an array is boolean.
- to_bool(image): Convert array to boolean mask.
- to_uint8(image): Convert array to uint8 [0, 255].
- fill_gaps(binary, area_threshold=50, radius=1, connectivity=2): Fill holes + optional closing.
- remove_noise(binary, min_size=20, connectivity=2): Remove small components.
- noise_and_gaps(img): remove_noise ∘ fill_gaps, returns uint8 mask.
- filter_contours_min_area(contours): Keep contours above min area.
- simplify_contours(contours, epsilon=0.5): RDP polygon simplification.
- auto_threshold_strategy(img, threshold): Apply named threshold + cleanup.
- channel_custom_threshold(channel, threshold): One channel → (mask, yolo_polygons).
- channel_auto_threshold(channel): Use default threshold from settings.
- channel_custom_segment(json_path, ratio): Load/normalize COCO polygons to YOLO.

Notes
-----
- Images are treated as uint8 unless explicitly stated; call `to_uint8` as needed.
- Areas/epsilons are in **pixels**.
- Returned YOLO polygons are **flattened, normalized** lists [x1, y1, ..., xn, yn] with
  coordinates in [0, 1] relative to the (square) target size from settings.

Rationale
---------
- Keep annotation steps deterministic and dataset-portable.
- Prefer simple, robust morphology
"""

import os
import cv2
import json
import scipy
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import binary_closing
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
import hfinder.core.hfinder_log as HFinder_log
import hfinder.core.hfinder_folders as HFinder_folders
import hfinder.core.hfinder_settings as HFinder_settings
import hfinder.core.hfinder_geometry as HFinder_geometry
import hfinder.core.hfinder_imageinfo as HFinder_ImageInfo


def mask_to_polygons(mask,
                     mode: str = "subtract_holes",
                     simplify_rel: float = 0.0003,
                     simplify_min: float = 0.05,
                     simplify_max: float = 0.3,
                     min_vertices: int = 3):
    """
    Convert a binary mask (0/255) to polygons, with optional hole retention.

    Modes:
      - "subtract_holes": fill externals then subtract internal holes → one polygon per region.
      - "rings": keep explicit rings per region: [outer, hole1, hole2, ...].

    Simplification:
      - Uses Ramer–Douglas–Peucker via `cv2.approxPolyDP` with an epsilon that
        scales with perimeter: `eps = clip(simplify_rel * perimeter, simplify_min, simplify_max)`.

    :param mask: Binary image with foreground as nonzero (uint8 recommended).
    :type mask: np.ndarray
    :param mode: "subtract_holes" or "rings".
    :type mode: str
    :param simplify_rel: Relative epsilon fraction of perimeter.
    :type simplify_rel: float
    :param simplify_min: Minimum absolute epsilon (px).
    :type simplify_min: float
    :param simplify_max: Maximum absolute epsilon (px).
    :type simplify_max: float
    :param min_vertices: Discard polygons with fewer vertices.
    :type min_vertices: int
    :return: If mode="rings": list of regions, each [outer, hole1, ...] (np.int32).
             Else: list of polygons (np.int32).
    :rtype: list[list[np.ndarray]] | list[np.ndarray]
    """
    m = (mask > 0).astype(np.uint8) * 255

    # Extract contours with 2-level hierarchy (externals + holes)
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return [] if mode == "subtract_holes" else []

    H = hier[0]
    h, w = m.shape[:2]

    def _simplify(c):
        # Perimeter-aware epsilon (bounded)
        L = cv2.arcLength(c, True)
        eps = np.clip(simplify_rel * L, simplify_min, simplify_max)
        return cv2.approxPolyDP(c, eps, True)

    # Minimum area for the local image, or global settings
    min_area = HFinder_ImageInfo.get("min_area_px")
    min_area = float(HFinder_settings.get("min_area_px")) if min_area is None else float(min_area)

    if mode == "rings":
        # Keep rings explicitly: one entry per region [outer, hole1, ...]
        regions = []
        for i, c in enumerate(contours):
            # parent == -1 → contour externe
            if H[i][3] != -1:
                continue
            if cv2.contourArea(c) < min_area:
                continue
            outer = _simplify(c)
            if len(outer) < min_vertices:
                continue

            # Children = holes
            holes = []
            ch = H[i][2]
            while ch != -1:
                cc = contours[ch]
                if cv2.contourArea(cc) >= min_area:
                    cc = _simplify(cc)
                    if len(cc) >= min_vertices:
                        holes.append(cc)
                ch = H[ch][0]
            regions.append([outer] + holes)
        return regions

    else:  # "subtract_holes"
        # Draw externals filled, then erase holes → one filled region per object
        filled = np.zeros_like(m)
        for i, c in enumerate(contours):
            if H[i][3] == -1 and cv2.contourArea(c) >= min_area:
                cv2.drawContours(filled, [c], -1, 255, -1)
                ch = H[i][2]
                while ch != -1:
                    if cv2.contourArea(contours[ch]) >= min_area:
                        cv2.drawContours(filled, [contours[ch]], -1, 0, -1)
                    ch = H[ch][0]

        final_contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        polys = []
        for c in final_contours:
            if cv2.contourArea(c) < min_area:
                continue
            c = _simplify(c)
            if len(c) >= min_vertices:
                polys.append(c.astype(np.int32))
        return polys



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
        - Otherwise, strictly positive values map to True (foreground),
          zeros/negatives to False (background).

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
        - other numeric dtypes follow scikit-image conversion rules

    :param image: Input array.
    :type image: np.ndarray
    :returns: uint8 array in the range 0–255.
    :rtype: np.ndarray
    """
    return skimage.util.img_as_ubyte(image)



def fill_gaps(binary, area_threshold=50, radius=1, connectivity=2):
    """
    Fill small black holes inside the white foreground and optionally smooth
    residual gaps by a binary closing.

    Pipeline:
        1) ``remove_small_holes`` fills cavities with area ≤ ``area_threshold``.
        2) If ``radius > 0``, apply ``binary_closing`` using a disk of given 
           ``radius`` (pixels).

    :param binary: Boolean mask (True = foreground). Convert with ``to_bool`` if needed.
    :type binary: np.ndarray
    :param area_threshold: Maximum hole area (in pixels) to be filled.
    :type area_threshold: int
    :param radius: Structuring element radius (in pixels) for the closing; set to 0 to disable.
    :type radius: int
    :param connectivity: Pixel connectivity for hole filling (2 → 8-connectivity in 2D).
    :type connectivity: int
    :returns: Boolean mask with small holes filled and (optionally) smoothed by closing.
    :rtype: np.ndarray
    :raises AssertionError: If ``binary`` is not boolean.
    """
    assert is_bool(binary), "(HFinder) Assert Failure: fill_gaps"
    filled_bool = remove_small_holes(binary,
                                     area_threshold=area_threshold,
                                     connectivity=connectivity)
    disk = skimage.morphology.disk(radius)
    closed_bool = binary_closing(filled_bool, footprint=disk)
    return closed_bool



# Alternative ideas to consider later:
#   disk1 = skimage.morphology.disk(1)
#   clean = skimage.morphology.opening(mask_bool, footprint=disk1)
#   clean = skimage.filters.median(mask_bool, footprint=disk1).astype(bool)
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



def filter_contours_min_area(contours):
    """
    Keep only contours whose area is at least the configured minimum.

    :param contours: OpenCV contours as returned by ``cv2.findContours``.
    :type contours: list[np.ndarray]
    :returns: Subset of contours with area >= ``HFinder_settings.get("min_area_px")``.
    :rtype: list[np.ndarray]
    """
    special_case = HFinder_ImageInfo.get("min_area_px")
    if special_case is None:
        # General setting used throughout the program
        min_area_px = HFinder_settings.get("min_area_px")
    else:
        # Local setting
        min_area_px = special_case
    return [c for c in contours if cv2.contourArea(c) >= float(min_area_px)]



# Ramer–Douglas–Peucker algorithm.
# Note: min_area_px ≈ π * (d_min / (2*s))**2 if we know the minimum object 
# diameter d_min (in µm) and pixel size s (in µm/px).
def simplify_contours(contours, epsilon=0.5):
    """
    Simplify contours with the Ramer–Douglas–Peucker algorithm.

    :param contours: OpenCV contours (each ``(N, 1, 2)`` array).
    :type contours: list[np.ndarray]
    :param epsilon: Approximation tolerance in pixels (0.5–2.0 px typical).
    :type epsilon: float
    :returns: Simplified contours (closed polygons preserved).
    :rtype: list[np.ndarray]
    :notes: Uses ``cv2.approxPolyDP(contour, epsilon, True)``.
    """
    return [cv2.approxPolyDP(c, epsilon, True) for c in contours]



def auto_threshold_strategy(img, threshold):
    """
    Apply a named thresholding method (scikit-image) and return the cleaned mask.

    Supported methods:
        - "isodata", "li", "otsu", "yen", "triangle", "mean"
        - "auto": heuristic based on skewness:
          if (mean - median) / (max + 1e-5) > 0.15 → "triangle", else "otsu".

    Post-processing:
        The binary mask (``img > thresh``) is cleaned via ``noise_and_gaps`` before return.

    :param img: Grayscale image (must be uint8).
    :type img: np.ndarray
    :param threshold: Method name (see above).
    :type threshold: str
    :returns: (numeric threshold, cleaned binary mask as uint8 0/255).
    :rtype: tuple[float, np.ndarray]
    :raises AssertionError: If ``img`` is not uint8.
    """
    assert img.dtype == np.uint8, "(HFinder) Assert Failure: auto_threshold_strategy"
    
    # Optional: exhaustive threshold exploration for debugging/papers
    if False: # FIXME: Insert this with an option
        base = HFinder_ImageInfo.get_current_base()
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
    elif threshold == "mean":
        thresh = skimage.filters.threshold_mean(img)  
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
        - If ``threshold`` is a string, use ``auto_threshold_strategy`` and post-process.
        - If ``threshold`` is numeric:
            * ``threshold`` < 1 → percentile (e.g., 0.9 = 90th percentile)
            * ``threshold`` ≥ 1 → absolute intensity threshold (0–255)
          Then threshold via OpenCV and post-process via ``noise_and_gaps``.

    Contours are extracted and converted to flattened YOLO-normalized polygons.

    :param channel: 2D single-channel image (uint8 recommended).
    :type channel: np.ndarray
    :param threshold: Method name, percentile fraction, or absolute threshold.
    :type threshold: str | float
    :returns: (cleaned binary mask, list of flattened YOLO-normalized polygons).
    :rtype: tuple[np.ndarray, list[list[float]]]
    """
    channel = to_uint8(channel)
    if isinstance(threshold, str):
        _, binary = auto_threshold_strategy(channel, threshold.lower())
    else:
        if threshold < 1:
            thresh_val = np.percentile(channel, threshold * 100)
        else:
            thresh_val = threshold

        _, binary = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY)
        binary = noise_and_gaps(binary)

    contours = mask_to_polygons(binary)

    # Filter out small contours
    contours = filter_contours_min_area(contours)
    
    # Convert to YOLO-normalized polygons
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
        - Get target size (``size``) from settings; fetch ratio from caller.
        - Rescale coordinates by ``ratio`` and then normalize to [0, 1] by dividing
          by ``size`` (square target).

    :param json_path: Path to the COCO-style JSON file.
    :type json_path: str
    :param ratio: Scale factor between the original image and the target size.
    :type ratio: float
    :returns: List of flattened polygons [x1, y1, ..., xn, yn] normalized to [0, 1].
    :rtype: list[list[float]]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    size = HFinder_settings.get("size")

    polygons = []
    for ann in data.get("annotations", []):
        if "segmentation" not in ann or not ann["segmentation"]:
            continue

        for seg in ann["segmentation"]:
            # Skip empty/odd-length lists; warn for traceability
            if not seg or len(seg) % 2 != 0:
                HFinder_log.warn(f"Invalid segmentation in {json_path}, " + \
                                 f"annotation id {ann.get('id')}")
                continue

            # First rescale to target pixels, then normalize by size (square
            flat = [seg[i] * ratio / size for i in range(len(seg))]
            polygons.append(flat)

    return polygons



