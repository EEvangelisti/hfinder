"""
Geometry utilities for HFinder.

This module provides helpers to:
- Convert OpenCV contours into YOLO-style normalized polygon annotations.
- Convert flat [x1, y1, x2, y2, ...] normalized coordinate lists back to
  integer pixel coordinates.

Public API
----------
- contours_to_yolo_polygons(contours, min_vertices=3, simplify_eps=None,
  max_points=None, as_flat=True): Convert contours to normalized polygons.
- flat_to_pts_xy(flat): Convert a flat normalized polygon to pixel points.

Notes
-----
- The normalization assumes that images have been resized to a square of side
  `size = HF_settings.get("size")`.
- Duplicate removal only removes consecutive duplicates.
- Sub-sampling uses linearly spaced indices to preserve shape coverage.
"""

import numpy as np
from hfinder.session import settings as HF_settings


def is_valid_image_format(img):
    """
    Check whether a NumPy array corresponds to a valid multi-channel or 
    z/t-stack image.

    Supported formats:
        - 3D multi-channel image: shape (C, H, W)
        - 4D z/t-stack: shape (Z, C, H, W)

    Constraints:
        - Fewer than 10 channels (C < 10)
        - Spatial dimensions strictly greater than 64×64

    :param img: Input image array.
    :type img: np.ndarray
    :return: True if the format is supported, False otherwise.
    :rtype: bool
    """
    valid_ndim = False
    if img.ndim == 4:
        valid_ndim = True
        _, c, h, w = img.shape
    elif img.ndim == 3:
        valid_ndim = True
        c, h, w = img.shape
    # Check dimensional validity, channel limit, and spatial size threshold
    return valid_ndim and c < 10 and h > 64 and w > 64



def clamp_box_xyxy(box, W, H):
    """
    Clamp an axis-aligned box to image bounds in (x1, y1, x2, y2) form.

    :param box: Box coordinates ``[x1, y1, x2, y2]`` (can be floats).
    :type box: list | tuple
    :param W: Image width in pixels.
    :type W: int
    :param H: Image height in pixels.
    :type H: int
    :returns: Clamped box ``[x1, y1, x2, y2]`` within ``[0..W-1] × [0..H-1]``.
    :rtype: list[int]
    """
    x1, y1, x2, y2 = box
    return [int(max(0, min(x1, W-1))),
            int(max(0, min(y1, H-1))),
            int(max(0, min(x2, W-1))),
            int(max(0, min(y2, H-1)))]



def contours_to_yolo_polygons(contours,
                              min_vertices=3,
                              simplify_eps=None,
                              max_points=None,
                              as_flat=True):
    """
    Convert OpenCV contours into YOLO-style normalized polygons.

    Each valid input contour of shape (N, 1, 2) is converted to a polygon whose
    coordinates are normalized to [0, 1] along both axes. Optional steps include:
    - Removal of consecutive duplicate vertices.
    - Polygon simplification (Douglas–Peucker algorithm).
    - Uniform sub-sampling to limit the number of points.

    **Assumptions**
      - Images have been resized to a *square* of side `size` pixels, where
        `size = HF_settings.get("size")`. Both x and y are divided by this
        same `size`. If your data are not square, adapt the normalization.

    :param contours: Iterable of OpenCV contours, each with shape (N, 1, 2).
    :type contours: Iterable[np.ndarray]
    :param min_vertices: Minimum number of vertices required to keep a polygon.
    :type min_vertices: int
    :param simplify_eps: Epsilon (in pixels) for cv2.approxPolyDP; None or <=0 disables.
    :type simplify_eps: float | None
    :param max_points: If provided, uniformly subsample each polygon to at most
        this many points.
    :type max_points: int | None
    :param as_flat: If True, return polygons as flat lists [x1, y1, x2, y2, ...];
        otherwise as (N, 2) arrays.
    :type as_flat: bool

    :return: List of polygons, each either a flattened list of floats or an
        (N, 2) float array, normalized to [0, 1].
    :rtype: list[list[float]] | list[np.ndarray]

    :notes:
        - Contours not matching the expected shape (N, 1, 2) are skipped.
        - Duplicate removal targets *consecutive* duplicates only.
        - Sub-sampling uses linearly spaced indices (including endpoints).
        - Output polygons do **not** repeat the first vertex at the end.
    """
    out = []

    # Fetch the (square) target side length for normalization
    size = HF_settings.get("size")

    for c in contours:
        # Skip invalid inputs: must be ndarray of shape (N, 1, 2)
        if not isinstance(c, np.ndarray) or c.ndim != 3 or c.shape[1:] != (1,2):
            continue

        # Collapse OpenCV's (N, 1, 2) → (N, 2) and cast to float
        arr = c.reshape(-1, 2).astype(float)

        # Remove consecutive duplicate vertices to avoid zero-length edges
        keep = [0]
        for i in range(1, len(arr)):
            if (arr[i] != arr[i-1]).any():
                keep.append(i)
        arr = arr[keep]
        
        # Skip if too few vertices remain after deduplication
        if arr.shape[0] < min_vertices:
            continue

        # Optional polygon simplification (Douglas–Peucker) in pixel space
        if simplify_eps and simplify_eps > 0:
            cc = arr.reshape(-1,1,2).astype(np.float32)
            cc = cv2.approxPolyDP(cc, epsilon=float(simplify_eps), closed=True)
            arr = cc.reshape(-1,2).astype(float)
            # Skip if simplification removed too many vertices
            if arr.shape[0] < min_vertices:
                continue

        # Optional uniform sub-sampling to limit vertex count
        if max_points and arr.shape[0] > max_points:
            idx = np.linspace(0, arr.shape[0] - 1, int(max_points), dtype=int)
            arr = arr[idx]

        # Normalize to [0, 1] assuming square canvas
        # Clip to avoid rounding issues
        arr[:,0] = np.clip(arr[:,0] / size, 0.0, 1.0)
        arr[:,1] = np.clip(arr[:,1] / size, 0.0, 1.0)

        # Append in requested format: flat list or (N, 2) array
        out.append(arr.flatten().tolist() if as_flat else arr)

    return out



def flat_to_pts_xy(flat):
    """
    Convert a flat normalized polygon to integer pixel coordinates.

    :param flat: Flat list [x1, y1, x2, y2, ...], normalized to [0, 1].
    :type flat: list[float]
    :param w: Target image width in pixels.
    :type w: int
    :param h: Target image height in pixels.
    :type h: int
    :return: Array of shape (N, 2) with integer pixel coordinates.
    :rtype: np.ndarray
    :raises AssertionError: If the flat list length is not even.
    """
    assert len(flat) % 2 == 0, "Polygon length must be even."
    # Multiply normalized coords by width/height to get absolute pixels.
    r = HF_settings.get("size")
    pts = [(int(flat[i] * r), int(flat[i+1] * r)) for i in range(0, len(flat), 2)]
    return np.asarray(pts, dtype=np.int32)
    


def bbox_xyxy_to_xywh(b):
    """
    Compute the area (in pixels²) of a bounding box [x1, y1, x2, y2].

    The result is max(0, x2−x1) * max(0, y2−y1), ensuring a non-negative area
    even if coordinates are partially inverted.

    :param b: Bounding box in xyxy format.
    :type b: list[float] | np.ndarray
    :return: Non-negative box area in pixel units.
    :rtype: float
    """
    x1, y1, x2, y2 = map(float, b)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]



def rescale_box_xyxy(box, scale_factor):
    """
    Rescale a bounding box from resized space back to original space.

    This function takes a bounding box in [x1, y1, x2, y2] format (resized space)
    and multiplies all values by the scale factor. The result is a bounding box
    expressed in the original TIFF coordinate system.

    :param box: Bounding box coordinates [x1, y1, x2, y2].
    :type box: list[float] | tuple[float, float, float, float]
    :param scale_factor: Factor to convert from resized to original coordinates
                         (usually 1.0 / resize_ratio).
    :type scale_factor: float
    :return: Rescaled bounding box in original coordinates.
    :rtype: list[float]
    """
    return [float(v) * scale_factor for v in box]



def rescale_seg_flat(seg, scale_factor):
    """
    Rescale a flattened polygon segmentation from resized space back to original space.

    This function takes a segmentation polygon represented as a flat list
    of alternating x and y coordinates, e.g. [x1, y1, x2, y2, ...], and multiplies
    all values by the scale factor. The result is a polygon expressed in the
    original TIFF coordinate system.

    :param seg: Flattened polygon segmentation (list of floats).
    :type seg: list[float]
    :param scale_factor: Factor to convert from resized to original coordinates
                         (usually 1.0 / resize_ratio).
    :type scale_factor: float
    :return: Rescaled polygon segmentation in original coordinates.
    :rtype: list[float]
    """
    return [float(v) * scale_factor for v in seg]



def poly_area_xy(poly):
    """
    Compute the polygon area (in pixels²) from a flattened [x1, y1, x2, y2, ...] list.
    Uses the shoelace formula. Degenerate inputs (fewer than 3 points) return 0.0.

    :param poly: Flattened polygon coordinates [x1, y1, x2, y2, ...].
    :type poly: list[float] | np.ndarray
    :return: Non-negative polygon area in pixel units.
    :rtype: float
    """
    if not poly or len(poly) < 6: 
        return 0.0
    it = iter(poly)
    pts = [(float(x), float(next(it))) for x in it]
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return 0.5 * abs(sum(x[i] * y[(i + 1) % len(pts)] - 
           x[(i + 1) % len(pts)] * y[i] for i in range(len(pts))))



def polys_from_xy(inst_xy):
    """
    Convert YOLO mask polygon(s) into COCO-style flattened lists.

    A YOLO result may store instance polygons either as:
      - a single (N,2) numpy array of x,y coordinates, or
      - a list of such arrays (for multiple disjoint polygons).

    This function normalizes the input into a list of flattened
    [x1, y1, x2, y2, ..., xn, yn] lists.

    :param inst_xy: A (N,2) array or list of arrays with polygon vertices.
    :type inst_xy: numpy.ndarray | list[numpy.ndarray] | None
    :return: A list of polygons, each represented as a flattened list of floats.
    :rtype: list[list[float]]
    """
    if inst_xy is None:
        return []
    polys = inst_xy if isinstance(inst_xy, (list, tuple)) else [inst_xy]
    flats = []
    for poly in polys:
        arr = np.asarray(poly)
        if arr.ndim != 2 or arr.shape[0] < 3:
            continue
        flats.append(arr.reshape(-1).tolist())
    return flats


       
