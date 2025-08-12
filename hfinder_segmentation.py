"""
hfinder_segmentation
Label-generation helpers for confocal images (dark background, grayscale).

Design choices
--------------
- 8-bit discipline: inputs are expected as uint8 [0–255]; conversions are 
  centralized in `to_uint8` / `to_bool`.
- Global thresholding via scikit-image or numeric value; `auto` selects Otsu or
  Triangle based on skewness.
- Morphological cleanup: remove small objects, then fill small holes + optional
  binary closing (`radius` in pixels).
- Optional instance splitting: watershed on the distance transform; seed density
  controlled by `min_distance`.
- Contour extraction: external contours with OpenCV; optional RDP simplification
  to cap vertex count.
- Area guard: discard contours smaller than `min_area_px`; no max-area by
  default (hyphae may be large).
- Outputs: uint8 masks (0/255) and YOLO-normalized polygons; intended for label
  creation, not model preprocessing.

Rationale
---------
- Keep annotations faithful to raw signal; use deterministic defaults and 
  dataset-level settings.
- Favor simple, reproducible steps that survive export changes and microscope
  sessions.
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
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings
import hfinder_geometry as HFinder_geometry


# ----------------

def mask_to_polygons(mask,
                     mode: str = "subtract_holes",
                     min_area: float = 60.0,
                     simplify_rel: float = 0.0003,
                     simplify_min: float = 0.05,
                     simplify_max: float = 0.3,
                     min_vertices: int = 3):
    """
    Convertit un masque binaire (0/255) en polygones.
    :param mode: "subtract_holes" (plein) ou "rings" (avec trous).
    :return: list[list[np.ndarray]] si mode="rings" (ext + trous),
             sinon list[np.ndarray] pour "subtract_holes".
    """
    m = (mask > 0).astype(np.uint8) * 255

    # Récupère contours + hiérarchie
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return [] if mode == "subtract_holes" else []

    H = hier[0]
    h, w = m.shape[:2]

    def _simplify(c):
        L = cv2.arcLength(c, True)
        eps = np.clip(simplify_rel * L, simplify_min, simplify_max)  # borne haute
        return cv2.approxPolyDP(c, eps, True)

    if mode == "rings":
        # Conserve anneaux : liste de [outer, hole1, hole2, ...] par région
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

            # enfants = trous
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
        # Dessine externes pleins puis efface les trous → un seul contour par région
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

# ----------------






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
    :retype: np.ndarray
    :raises AssertionError: If ``binary`` is not boolean.
    """
    assert is_bool(binary), "(HFinder) Assert Failure: fill_gaps"
    filled_bool = remove_small_holes(binary,
                                     area_threshold=area_threshold,
                                     connectivity=connectivity)
    disk = skimage.morphology.disk(radius)
    closed_bool = binary_closing(filled_bool, footprint=disk)
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



def split_touching_watershed(binary):
    """
    Split merged foreground regions using distance transform + watershed,
    then return one contour per resulting instance.

    Pipeline:
        1) Euclidean distance transform (EDT) on the binary mask.
        2) Seed extraction from local maxima (enforcing ``min_distance`` between peaks).
        3) Watershed on the negative distance restricted to the foreground.
        4) For each label, extract a single external OpenCV contour.

    :param binary: Binary mask (nonzero = foreground). Boolean is recommended.
    :type binary: np.ndarray
    :returns: List of OpenCV contours, each as an ``(N, 1, 2)`` integer array.
    :retype: list[np.ndarray]
    :notes: Uses a peak detector on the distance map (``peak_local_max``) to
            control seed separation via ``min_distance``.
    """
    binary = to_bool(binary)
    dist = scipy.ndimage.distance_transform_edt(binary)
    min_distance = HFinder_settings.get("min_distance")
    coords = peak_local_max(dist, min_distance=min_distance,
                            labels=binary.astype(bool), exclude_border=False)
    peaks = np.zeros_like(dist, dtype=bool)
    if coords.size:
        peaks[tuple(coords.T)] = True
    markers = scipy.ndimage.label(peaks)[0]
    labels = skimage.segmentation.watershed(-dist, markers, mask=binary)
    labels = np.asarray(labels)
    contours = []
    for lab in range(1, int(labels.max()) + 1):
        m = (labels == lab).astype(np.uint8) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cs)
    return contours


def filter_contours_min_area(contours):
    """
    Keep only contours whose area is at least the configured minimum.

    :param contours: OpenCV contours as returned by ``cv2.findContours``.
    :type contours: list[np.ndarray]
    :returns: Subset of contours with area >= ``HFinder_settings.get("min_area_px")``.
    :retype: list[np.ndarray]
    """
    min_area_px = HFinder_settings.get("min_area_px")
    return [c for c in contours if cv2.contourArea(c) >= float(min_area_px)]



def simplify_contours(contours, epsilon=0.5):
    """
    Simplify contours with the Ramer–Douglas–Peucker algorithm.

    :param contours: OpenCV contours (each ``(N, 1, 2)`` array).
    :type contours: list[np.ndarray]
    :param epsilon: Approximation tolerance in pixels (0.5–2.0 px typical).
    :type epsilon: float
    :returns: Simplified contours (closed polygons preserved).
    :retype: list[np.ndarray]
    :notes: Uses ``cv2.approxPolyDP(contour, epsilon, True)``.
    """
    return [cv2.approxPolyDP(c, epsilon, True) for c in contours]



# Ramer–Douglas–Peucker algorithm.
# Note: min_area_px ≈ π * (d_min / (2*s))**2 if we know the minimum object 
# diameter d_min (in µm) and pixel size s (in µm/px).
# Since OpenCV 3.2, findContours() no longer modifies the source image but 
# returns a modified image as the first of three return parameters.
def find_contours(mask):
    """
    Find external contours in a binary mask using OpenCV.

    :param binary: Binary image (uint8, values in {0, 255}). Convert with
                   ``to_uint8`` if the input is boolean.
    :type binary: np.ndarray
    :returns: External contours as ``(N, 1, 2)`` integer arrays.
    :retype: list[np.ndarray]
    :notes: In OpenCV ≥ 3.2 the source image is not modified by
            ``cv2.findContours``; for older versions the function used to
            alter the input and returned three values.
    """
    polygons = mask_to_polygons(mask, mode="subtract_holes")
    return polygons


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

    # Find contours from the binary mask, either after watershed, or directly
    use_watershed = HFinder_settings.get("watershed")
    # Not used by default (due to rather messy output...)
    if use_watershed:
        contours = split_touching_watershed(binary)
    else:
        contours = find_contours(binary)
    # Simplify contours
    #contours = simplify_contours(contours)
    # Filter out small contours
    contours = filter_contours_min_area(contours)
    
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

    size = HFinder_settings.get("size")

    polygons = []
    for ann in data.get("annotations", []):
        if "segmentation" not in ann or not ann["segmentation"]:
            continue

        for seg in ann["segmentation"]:
            if not seg or len(seg) % 2 != 0:
                HFinder_log.warn(f"Invalid segmentation in {json_path}, " + \
                                 f"annotation id {ann.get('id')}")
                continue

            flat = [seg[i] * ratio / size for i in range(len(seg))]
            polygons.append(flat)

    return polygons



