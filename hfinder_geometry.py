import sys
import math
import numpy as np
import hfinder_settings as HFinder_settings



def contours_to_yolo_polygons(contours,
                              min_vertices=3,
                              simplify_eps=None,
                              max_points=None,
                              as_flat=True):
    """
    Convert OpenCV contours into YOLO-style normalized polygons.

    Each valid input contour of shape (N, 1, 2) is converted to a polygon whose
    coordinates are normalized to [0, 1] along both axes. Optional steps include
    removal of consecutive duplicate vertices, polygon simplification
    (Douglas–Peucker), and uniform sub-sampling to limit the number of points.

    **Assumptions**
      - Images have been resized to a *square* of side `size` pixels, where
        `size = HFinder_settings.get("size")`. Both x and y are divided by this
        same `size`. If your data are not square, adapt the normalization.

    :param contours: Iterable of OpenCV contours, each with shape (N, 1, 2).
    :type contours: Iterable[numpy.ndarray]
    :param min_vertices: Minimum number of vertices required to keep a polygon.
    :type min_vertices: int
    :param simplify_eps: Epsilon (in pixels) for cv2.approxPolyDP; None or <=0 
        disables simplification.
    :type simplify_eps: float | None
    :param max_points: If provided, uniformly subsample each polygon to at most
        this many points.
    :type max_points: int | None
    :param as_flat: If True, return polygons as flat lists [x1, y1, x2, y2, ...]; 
        otherwise as (N, 2) arrays.
    :type as_flat: bool

    :returns: List of polygons, each either a flattened list of floats or an 
        (N, 2) float array, normalized to [0, 1].
    :rtype: list[list[float]] | list[numpy.ndarray]

    :notes:
        - Contours not matching the expected shape (N, 1, 2) are silently skipped.
        - Duplicate removal targets *consecutive* duplicates only; it does not de-duplicate globally.
        - Sub-sampling uses linearly spaced indices across the vertex list (including endpoints).
        - The returned polygons do **not** repeat the first vertex at the end.

    """
    out = []

    # Fetch the (square) target side length used for normalization.
    size = HFinder_settings.get("size")

    for c in contours:
        # Skip anything that is not a NumPy array of shape (N, 1, 2).
        if not isinstance(c, np.ndarray) or c.ndim != 3 or c.shape[1:] != (1,2):
            continue

        # Collapse (N, 1, 2) -> (N, 2) and ensure floating-point arithmetic.
        # Contours are expected in OpenCV format (N, 1, 2) — N vertices, each 
        # wrapped in an extra singleton dimension before the (x, y) coordinate
        # pair
        arr = c.reshape(-1, 2).astype(float)

        # Remove consecutive duplicate vertices to avoid zero-length edges.
        keep = [0]
        for i in range(1, len(arr)):
            if (arr[i] != arr[i-1]).any():
                keep.append(i)
        arr = arr[keep]
        
        # Discard if too few vertices remain.
        if arr.shape[0] < min_vertices:
            continue

        # Optional polygon simplification (Douglas–Peucker) in pixel space.
        if simplify_eps and simplify_eps > 0:
            cc = arr.reshape(-1,1,2).astype(np.float32)
            cc = cv2.approxPolyDP(cc, epsilon=float(simplify_eps), closed=True)
            arr = cc.reshape(-1,2).astype(float)
            # Discard if simplification produced too few vertices.
            if arr.shape[0] < min_vertices:
                continue

        # Optional uniform sub-sampling to limit the vertex count.
        if max_points and arr.shape[0] > max_points:
            idx = np.linspace(0, arr.shape[0] - 1, int(max_points), dtype=int)
            arr = arr[idx]

        # Normalize coordinates to [0, 1] assuming a square canvas of side `size`.
        # Clip for numerical safety after division.
        arr[:,0] = np.clip(arr[:,0] / size, 0.0, 1.0)
        arr[:,1] = np.clip(arr[:,1] / size, 0.0, 1.0)

        # Append either a flat [x1, y1, ...] list or the (N, 2) array.
        out.append(arr.flatten().tolist() if as_flat else arr)

    return out



def flat_to_pts_xy(flat, w, h):
    """flat: [x1,y1,x2,y2,...] normalisés → ndarray (N,2) en pixels int32."""
    assert len(flat) % 2 == 0, "Polygon length must be even."
    pts = [(int(flat[i] * w), int(flat[i+1] * h)) for i in range(0, len(flat), 2)]
    return np.asarray(pts, dtype=np.int32)
