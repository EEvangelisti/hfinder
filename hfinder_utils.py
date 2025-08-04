import numpy as np
from matplotlib.colors import hsv_to_rgb
import hfinder_settings as HFinder_settings



def colorize_with_hue(frame, hue):
    norm = frame.astype(np.float32)
    norm /= norm.max() if norm.max() > 0 else 1

    h = np.full_like(norm, hue)
    s = np.ones_like(norm)
    v = norm

    hsv = np.stack([h, s, v], axis=-1)  # shape (H, W, 3)
    rgb = hsv_to_rgb(hsv)
    return rgb



def contours_to_yolo_polygons(contours):
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



def is_valid_image_format(img):
    # Time series of z-stacks.
    if img.ndim == 4:
        _, c, h, w = img.shape
        return c < 10 and h > 64 and w > 64
    # Standard multichannel TIFF.
    elif img.ndim == 3:
        c, h, w = img.shape
        return c < 10 and h > 64 and w > 64
    # Other formats we cannot handle.
    else:
        return False



def save_yolo_segmentation_label(file_path, polygons, class_ids):
    """
    Writes YOLOv8-style segmentation labels to a .txt file, with one line per 
    polygon, using normalized coordinates.

    Parameters:
        file_path (str): Path to the output label file.
        polygons (list[tuple[str, list[tuple[float, float]]]]): List of tuples 
        (class_name, polygon).
        class_ids (dict[str, int]): Mapping from class names to YOLO integer IDs.
    """
    with open(file_path, "w") as f:
        for class_name, poly in polygons:
            if class_name not in class_ids:
                continue
            class_id = class_ids[class_name]
            poly = [coord for point in poly for coord in point]
            line = [str(class_id)] + [f"{x:.6f}" for x in poly]
            f.write(" ".join(line) + "\n")

