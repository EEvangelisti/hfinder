import sys
import math
import numpy as np
import hfinder_settings as HFinder_settings


def contours_to_yolo_polygons_old(contours):
    """
    Converts a list of OpenCV contours into YOLO-style normalized polygon 
    annotations. Each contour is expected to be an array of shape (N, 1, 2) or 
    (N, 2), representing N (x, y) points. The function filters out invalid or 
    degenerate contours (e.g. fewer than 3 points), normalizes the coordinates 
    to the [0, 1] range using the target image size, and flattens each polygon 
    into a single list of coordinates.

    Returns:
        List of polygons, where each polygon is represented as a flat list:
        [x1, y1, x2, y2, ..., xn, yn], with all coordinates normalized to [0, 1].
    """
    yolo_polygons = []
    w, h = HFinder_settings.get("target_size")

    for contour in contours:
        # Remove redundant dimensions (e.g., shape (N,1,2) -> (N,2))
        polygon = contour.squeeze()
        # Skip if not a valid polygon (less than 3 points or bad shape)
        if len(polygon.shape) != 2 or polygon.shape[0] < 3:
            continue
        # Normalize polygon coordinates to [0,1] range
        norm_poly = [(x/w, y/h) for x, y in polygon]
        # Flatten list of (x,y) pairs to a single list [x1, y1, x2, y2, ..., xn, yn]
        flat_poly = [coord for point in norm_poly for coord in point]
        # Store the flattened polygon
        yolo_polygons.append(flat_poly)
    return yolo_polygons



def contours_to_yolo_polygons_info(contours):

    print(f"Type global de contours: {type(contours)}")
    try:
        print(f"Nombre de contours: {len(contours)}")
    except Exception as e:
        print(f"Impossible de calculer len(contours) : {e}")

    for i, contour in enumerate(contours):
        print(f"\n--- Contour {i} ---")
        print(f"Type: {type(contour)}")
        try:
            arr = np.asarray(contour)
            print(f"np.asarray shape: {arr.shape}, dtype: {arr.dtype}, ndim: {arr.ndim}")
            print(f"Premier élément brut: {contour[0] if len(contour) else 'vide'}")
        except Exception as e:
            print(f"Erreur conversion numpy: {e}")
        print(f"Représentation brute: {repr(contour)[:300]}{'...' if len(repr(contour))>300 else ''}")

    sys.exit("Fin debug_contours_info : inspection terminée.")



def contours_to_yolo_polygons(contours,
                              min_vertices=3,
                              simplify_eps=None,   # en px (None = pas de simplification)
                              max_points=None,     # limite de points par polygone
                              as_flat=True):       # True: [x1,y1,...]; False: (N,2)
    """Contours OpenCV (N,1,2) → polygones YOLO normalisés [0,1]."""
    out = []

    for c in contours:
        if not isinstance(c, np.ndarray) or c.ndim != 3 or c.shape[1:] != (1,2):
            continue  # on ignore tout ce qui n'est pas (N,1,2)
        arr = c.reshape(-1, 2).astype(float)

        # supprimer doublons consécutifs
        keep = [0]
        for i in range(1, len(arr)):
            if (arr[i] != arr[i-1]).any():
                keep.append(i)
        arr = arr[keep]
        if arr.shape[0] < min_vertices:
            continue

        # simplification optionnelle (sur pixels)
        if simplify_eps and simplify_eps > 0:
            cc = arr.reshape(-1,1,2).astype(np.float32)
            cc = cv2.approxPolyDP(cc, epsilon=float(simplify_eps), closed=True)
            arr = cc.reshape(-1,2).astype(float)
            if arr.shape[0] < min_vertices:
                continue

        # sous-échantillonnage optionnel
        if max_points and arr.shape[0] > max_points:
            idx = np.linspace(0, arr.shape[0]-1, int(max_points), dtype=int)
            arr = arr[idx]

        # normalisation [0,1]
        arr[:,0] = np.clip(arr[:,0] / 640, 0.0, 1.0)
        arr[:,1] = np.clip(arr[:,1] / 640, 0.0, 1.0)

        out.append(arr.flatten().tolist() if as_flat else arr)
    return out


def flat_to_pts_xy(flat, w, h):
    """flat: [x1,y1,x2,y2,...] normalisés → ndarray (N,2) en pixels int32."""
    assert len(flat) % 2 == 0, "Polygon length must be even."
    pts = [(int(flat[i] * w), int(flat[i+1] * h)) for i in range(0, len(flat), 2)]
    return np.asarray(pts, dtype=np.int32)
