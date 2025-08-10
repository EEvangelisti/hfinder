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

import cv2
import json
import numpy as np
import hfinder_log as HFinder_log
import hfinder_settings as HFinder_settings
import hfinder_geometry as HFinder_geometry



def isodata_threshold(image, nbins=256, tol=0.5, max_iter=256, use_histogram=True):
    """
    Calcule le seuil IsoData (Ridler–Calvard) d'une image NumPy.

    Paramètres
    ----------
    image : np.ndarray
        Image en niveaux de gris (H x W) ou tableau quelconque convertible en float.
        Les NaN sont ignorés.
    nbins : int, optionnel
        Nombre de classes pour l'histogramme (si use_histogram=True). Par défaut 256.
    tol : float, optionnel
        Tolérance d'arrêt sur la variation du seuil entre deux itérations. Par défaut 0.5 (en unités d'intensité).
    max_iter : int, optionnel
        Nombre maximal d'itérations. Par défaut 256.
    use_histogram : bool, optionnel
        Si True, itère sur l'histogramme (rapide et stable). Sinon, itère directement sur les pixels.

    Retour
    ------
    float
        Seuil IsoData.
    """
    x = np.asarray(image, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("Image vide (ou uniquement des NaN).")

    if not use_histogram:
        t = x.mean()
        for _ in range(max_iter):
            low = x[x <= t]
            high = x[x >  t]
            if low.size == 0 or high.size == 0:
                # cas dégénéré : une classe vide -> on renvoie le seuil courant
                return float(t)
            t_new = 0.5 * (low.mean() + high.mean())
            if abs(t_new - t) < tol:
                return float(t_new)
            t = t_new
        return float(t)

    # Version histogramme (plus rapide, comportement très proche d'ImageJ)
    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        return float(xmin)

    hist, edges = np.histogram(x, bins=nbins, range=(xmin, xmax))
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = hist.astype(float)

    if counts.sum() == 0:
        return float(centers[0])

    # initialisation : moyenne pondérée
    t = (counts @ centers) / counts.sum()

    for _ in range(max_iter):
        mask_low  = centers <= t
        mask_high = ~mask_low

        w_low  = counts[mask_low].sum()
        w_high = counts[mask_high].sum()

        if w_low == 0 or w_high == 0:
            return float(t)

        m_low  = (counts[mask_low]  @ centers[mask_low])  / w_low
        m_high = (counts[mask_high] @ centers[mask_high]) / w_high

        t_new = 0.5 * (m_low + m_high)

        if abs(t_new - t) < tol:
            return float(t_new)
        t = t_new

    return float(t)



def isodata_binarize(image, **kwargs):
    """
    Binarise l'image avec IsoData. Retourne (binaire, seuil).
    """
    t = isodata_threshold(image, **kwargs)
    return t, (np.asarray(image, dtype=float) > t)



def auto_threshold_strategy(img, threshold):
    """
    Automatically selects the best thresholding method (OTSU or Triangle) 
    depending on whether the signal is minority or not.
    
    :param img: Grayscale (uint8) image
    :type img: np.ndarray
    :param threshold: thresholding function (auto, otsu, triangle)
    :type threshold: str
    :returns: Binary image after thresholding
    :retype: np.ndarray 
    """
    assert img.dtype == np.uint8, "Input image must be uint8"
    
    if threshold == "isodata":
        return isodata_binarize(img)
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
    auto_threshold = HFinder_settings.get("default_auto_threshold")
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



