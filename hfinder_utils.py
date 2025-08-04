import os
import yaml
import numpy as np
from itertools import combinations
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



def write_yolo_yaml(class_ids, folder_tree):
    """
    Generate and save a YOLO-compatible dataset YAML file. This function creates
    a `dataset.yaml` file describing the training and validation dataset 
    structure for YOLOv8, including class names, number of classes, and paths
    to training/validation images. The file is written to `dataset/dataset.yaml`
    within the project's root directory.

    Parameters:
        class_ids (dict): A dictionary mapping class names (str) to class 
                          indices (int). Example: {"cell": 0, "noise": 1}
        folder_tree (dict): A dictionary representing folder paths used in the 
                            project. Must include the key "root" with the path 
                            to the root directory.

    Side effects:
        - Writes a `dataset.yaml` file to the dataset directory.
        - Updates `folder_tree` with a new key `"dataset/yaml"` pointing to the
          YAML path.
    """
    data = {
        "path": HFinder_folders.get_root(),
        "train": HFinder_folders.get_image_train_dir(),
        "val": HFinder_folders.get_image_val_dir(),
        "nc": len(class_ids),
        "names": [name for name, idx in sorted(class_ids.items(), key=lambda x: x[1])]
    }

    yaml_path = os.path.join(HFinder_folders.get_dataset_dir(), "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    HFinder_folders.set_subtree(folder_tree, "dataset/yaml", yaml_path)






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



def power_set(channels, n, c):
    if n == 1:
        s = list(channels)
        return [combo for r in range(1, 4) for combo in combinations(s, r)]
    
    all_combos = []
    for stack_index in range(n):
        # Canaux de ce stack présents dans `channels`
        group = [ch for ch in channels if (ch - 1) // c == stack_index]
        for r in range(1, 4):
            all_combos.extend(combinations(group, r))
    return all_combos


