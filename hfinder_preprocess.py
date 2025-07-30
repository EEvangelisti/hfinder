import os
import cv2
import json
import yaml
import random
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from collections import defaultdict
from itertools import chain, combinations
from PIL import Image
from numpy import stack, uint8
from matplotlib.colors import hsv_to_rgb
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings



def load_class_definitions():
    """
    Extract class names from all JSON files in `data_dir/classes/`, assign them integer IDs.

    Returns:
        dict: Mapping from class name to class ID (int), e.g., {"hyphae": 0, "nuclei": 1}
    """
    class_dir = os.path.join(HFinder_settings.get("tiff_dir"), "classes")
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"No such directory: {class_dir}")

    class_files = sorted(glob(os.path.join(class_dir, "*.json")))
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in class_files]
    return {name: idx for idx, name in enumerate(class_names)}



def write_yolo_yaml(class_ids, folder_tree):
    root = folder_tree["root"]
    train_dir = os.path.join(root, "dataset", "images", "train")
    val_dir = os.path.join(root, "dataset", "images", "val")

    data = {
        "path": root,
        "train": train_dir,
        "val": val_dir,
        "nc": len(class_ids),
        "names": [name for name, idx in sorted(class_ids.items(), key=lambda x: x[1])]
    }

    output_path = os.path.join(root, "dataset", "dataset.yaml")
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    HFinder_folders.set_subtree(folder_tree, "dataset/yaml", output_path)



def load_image_class_mappings():
    """
    Create a mapping: image → class → extraction instructions.

    Returns:
        dict: image filename → {class_name: {"channel":..., "segment":...} or -1}
    """
    class_dir = os.path.join(HFinder_settings.get("tiff_dir"), "classes")
    if not os.path.isdir(class_dir):
        HFinder_log.fail(f"No such directory: {class_dir}", exit_code=4)

    class_files = sorted(glob(os.path.join(class_dir, "*.json")))
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in class_files]

    # Step 1: build class-wise image mappings
    per_class_maps = {}
    for json_file in class_files:
        class_name = os.path.splitext(os.path.basename(json_file))[0]
        with open(json_file, "r") as f:
            raw_map = json.load(f)
        per_class_maps[class_name] = {}
        for img, val in raw_map.items():
            if isinstance(val, int):
                per_class_maps[class_name][img] = {"channel": val}
            elif isinstance(val, dict) and "channel" in val:
                per_class_maps[class_name][img] = val
            else:
                HFinder_log.fail(f"Invalid annotation for image {img} in class {class_name}",
                                 exit_code=HFinder_log.EXIT_INVALID_ANNOTATION)

    # Step 2: merge into image-wise mapping
    all_images = set()
    for d in per_class_maps.values():
        all_images.update(d.keys())

    final_map = {}
    for img in sorted(all_images):
        final_map[img] = {}
        for class_name in class_names:
            if img in per_class_maps[class_name]:
                final_map[img][class_name] = per_class_maps[class_name][img]
            else:
                final_map[img][class_name] = -1
    return final_map



def channel_auto_threshold(channel):
    auto_threshold = HFinder_settings.get("default_auto_threshold")
    return channel_custom_threshold(channel, auto_threshold)



def channel_custom_threshold(channel, threshold):
    w, h = HFinder_settings.get("target_size")
    thresh_val = np.percentile(channel, threshold)
    _, binary = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Prepare list to store polygons in YOLO-style normalized coordinates
    yolo_polygons = []
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
    return binary, yolo_polygons



def channel_custom_segment(json_path, ratio):
    with open(json_path, "r") as f:
        data = json.load(f)

    w, h = HFinder_settings.get("target_size")

    polygons = []
    for ann in data.get("annotations", []):
        if "segmentation" not in ann or not ann["segmentation"]:
            continue

        for seg in ann["segmentation"]:
            if not seg or len(seg) % 2 != 0:
                HFinder_log.warn(f"Invalid segmentation in {json_path}, annotation id {ann.get('id')}")
                continue

            # seg est une liste plate : [x0, y0, x1, y1, ...]
            # On la normalise directement
            flat = [seg[i] * ratio / w if i % 2 == 0 else seg[i] * ratio / h for i in range(len(seg))]
            polygons.append(flat)

    return polygons



def prepare_class_inputs(folder_tree, base, channels, class_instructions, ratio):
    """
    Prepares class-specific inputs for a given image based on user-defined instructions.

    Args:
        image_path (str): Path to the original image (e.g., TIFF).
        class_instructions (dict): Dictionary mapping class names to processing instructions.
            The instruction can be:
                - an integer → auto-thresholding on that channel
                - a dict with a "threshold" → custom thresholding
                - a dict with a "segment" → use manual segmentation mask (COCO JSON)

    Returns:
        dict: Mapping from class name to result of corresponding processing function.
    """
    results = defaultdict(list)

    for class_name, instr in class_instructions.items():
        if instr == -1:
            continue
    
        ch = instr["channel"]

        if "threshold" in instr:
            binary, polygons = channel_custom_threshold(channels[ch], instr["threshold"])
            results[class_name].append((class_name, polygons))
            binary_output = os.path.join(folder_tree["root"], f"dataset/masks/{base}_{class_name}_mask.png")
            plt.imsave(binary_output, binary, cmap='gray')

        elif "segment" in instr:
            json_path = os.path.join(HFinder_settings.get("tiff_dir"), instr["segment"])
            polygons = channel_custom_segment(json_path, ratio)
            results[ch].append((class_name, polygons))

        else:
            binary, polygons = channel_auto_threshold(channels[ch])
            results[ch].append((class_name, polygons))
            binary_output = os.path.join(folder_tree["root"], f"dataset/masks/{base}_{class_name}_mask.png")
            plt.imsave(binary_output, binary, cmap='gray')

    return results



def is_channel_first(img):
    if img.ndim != 3:
        return False
    c, h, w = img.shape
    return c < 10 and h > 64 and w > 64



def resize_multichannel_image(img):
    """
    Resize a multichannel image (C, H, W) to the given size using bilinear 
    interpolation per channel.

    Args:
        img (np.ndarray): Input image of shape (C, H, W).
        target_size (tuple): (height, width) desired.

    Returns:
        np.ndarray: Resized image of shape (C, target_height, target_width).
    """

    target_size = HFinder_settings.get("target_size")
    c, h, w = img.shape
    resized = np.empty((c, *target_size), dtype=img.dtype)
    for i in range(c):
        resized[i] = cv2.resize(img[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    ratios = tuple(x / w for x in target_size)
    assert(ratios[0] == ratios[1]) # TODO: Remove this?
    return {i + 1: resized[i] for i in range(c)}, ratios[0]



def split_train_val(image_paths, train_ratio=0.8, seed=42):
    image_paths = image_paths.copy()
    random.Random(seed).shuffle(image_paths)
    n_train = int(len(image_paths) * train_ratio)
    return dict([(path, True)  for path in image_paths[:n_train]] + \
                [(path, False) for path in image_paths[n_train:]])



def colorize_with_hue(frame, hue):
    norm = frame.astype(np.float32)
    norm /= norm.max() if norm.max() > 0 else 1

    h = np.full_like(norm, hue)
    s = np.ones_like(norm)
    v = norm

    hsv = np.stack([h, s, v], axis=-1)  # shape (H, W, 3)
    rgb = hsv_to_rgb(hsv)
    return rgb



def generate_dataset(folder_tree, base, split_flag, channels, polygons_per_channel):
    def power_set(iterable):
        """Sous-ensembles non vides jusqu'à 3 éléments."""
        s = list(iterable)
        return [combo for r in range(1, 4) for combo in combinations(s, r)]

    def compose_hue_fusion(channels, selected_channels):
        """
        Compose an RGB image by blending each selected channel with a random hue.
        """
        h, w = next(iter(channels.values())).shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for ch in selected_channels:
            frame = channels[ch]
            hue = np.random.rand()  # [0, 1)
            colored = colorize_with_hue(frame, hue)
            rgb += colored  # additive mixing

        # Clip to [0, 1] in case of saturation, then scale to [0, 255]
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)

    def convert_polygon_to_yolo_bbox(polygon, img_w, img_h):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_center = (min(xs) + max(xs)) / 2 / img_w
        y_center = (min(ys) + max(ys)) / 2 / img_h
        box_w = (max(xs) - min(xs)) / img_w
        box_h = (max(ys) - min(ys)) / img_h
        return x_center, y_center, box_w, box_h

    def make_filename(base, combo):
        return f"{os.path.splitext(base)[0]}_" + "_".join(map(str, combo)) + ".jpg"

    def save_image_as_jpg(img_rgb, path):
        Image.fromarray(img_rgb).save(path, "JPEG")

    def save_annotations_yolo_format(path, polygons, img_w, img_h, class_ids):
        with open(path, "w") as f:
            for class_name, poly in polygons:
                if class_name not in class_ids:
                    continue
                class_id = class_ids[class_name]
                bbox = convert_polygon_to_yolo_bbox(poly, img_w, img_h)
                f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n")

    # Détermination du sous-dossier (train/val)
    split = "train" if split_flag else "val"
    img_dir = os.path.join(folder_tree["root"], "dataset", "images", split)
    lbl_dir = os.path.join(folder_tree["root"], "dataset", "labels", split)

    # Chargement des identifiants de classes
    class_ids = load_class_definitions()  # suppose définie ailleurs
    target_size = HFinder_settings.get("target_size")
    img_h, img_w = target_size

    for combo in power_set(polygons_per_channel.keys()):
        filename = make_filename(base, combo)
        img_rgb = compose_hue_fusion(channels, combo)
        img_path = os.path.join(img_dir, filename)
        save_image_as_jpg(img_rgb, img_path)

        # Annotations combinées
        annotations = list(chain.from_iterable(polygons_per_channel.get(ch, []) for ch in combo))
        if annotations:
            label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
            save_annotations_yolo_format(label_path, annotations, img_w, img_h, class_ids)



def generate_training_dataset(folder_tree):

    # Retrieve images and assign to training or validation subsets
    data_dir = HFinder_settings.get("tiff_dir")
    image_paths = sorted(glob(os.path.join(data_dir, "*.tif")))
    split_table = split_train_val(image_paths)
    
    # Loads classes and preprocessing instructions.
    class_ids = load_class_definitions()
    write_yolo_yaml(class_ids, folder_tree)
    class_instructions = load_image_class_mappings()
    target_size = HFinder_settings.get("target_size")

    for img_path in image_paths:

        # Check whether there is something to detect on the image.
        # TODO: Maybe include these as noise-only images?
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]
        if img_name not in class_instructions:
            HFinder_log.warn(f"Skipping file {img_name} - no annotations")
            continue  # skip images not listed

        # Ensure this is a proper TIFF file.
        # FIXME: handle z-stacks and time series.
        img = tifffile.imread(img_path)
        if not is_channel_first(img):
            HFinder_log.warn(f"Skipping file {img_name} - not a multichannel image")
            continue

        # Resize image and split channels.
        channels, ratio = resize_multichannel_image(img)   
        
        # Retrieve polygons for each class and generate dataset.  
        polygons_per_channel = prepare_class_inputs(folder_tree, base, channels, class_instructions[img_name], ratio)       
        generate_dataset(folder_tree, base, split_table[img_path], channels, polygons_per_channel)

