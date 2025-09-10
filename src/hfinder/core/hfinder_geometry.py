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
from hfinder.core import hf_settings as HF_settings



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
