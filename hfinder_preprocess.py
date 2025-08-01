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
import hfinder_palette as HFinder_palette
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
    """
    Generate and save a YOLO-compatible dataset YAML file.

    This function creates a `dataset.yaml` file describing the training and validation
    dataset structure for YOLOv8, including class names, number of classes, and paths
    to training/validation images. The file is written to `dataset/dataset.yaml`
    within the project's root directory.

    Parameters:
        class_ids (dict): A dictionary mapping class names (str) to class indices (int).
                          Example: {"cell": 0, "noise": 1}
        folder_tree (dict): A dictionary representing folder paths used in the project.
                            Must include the key "root" with the path to the root directory.

    Side effects:
        - Writes a `dataset.yaml` file to the dataset directory.
        - Updates `folder_tree` with a new key `"dataset/yaml"` pointing to the YAML path.

    Output YAML format:
        path: <root directory>
        train: <path to training images>
        val: <path to validation images>
        nc: <number of classes>
        names: <list of class names ordered by index>
    """

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
    Load and consolidate image-to-class channel mappings from JSON annotation files.

    This function reads all JSON files in the `classes/` subdirectory of the 
    TIFF data directory (as specified in the HFinder settings). Each file must 
    be named after a class and contain a mapping from image filenames to either:
      - an integer (representing the channel for that class in the image), or
      - a dictionary containing at least the key `"channel"`.

    The function first builds per-class mappings, then merges them into a 
    unified image-wise map where, for each image, every class is listed with 
    its associated channel mapping or `-1` if absent.

    Returns:
        dict: A nested dictionary of the form:
              {
                  "image_name_1": {
                      "class_A": {"channel": X},
                      "class_B": -1,
                      ...
                  },
                  ...
              }

    Raises:
        - Logs a fatal error and exits if:
          - The `classes/` directory does not exist.
          - Any annotation is malformed (missing `"channel"` key or incorrect format).

    Dependencies:
        - Uses `HFinder_settings.get("tiff_dir")` to locate the base data directory.
        - Expects each class to be represented by a single JSON file in `classes/`.
        - Logs errors via `HFinder_log.fail`.

    Example expected input format (inside a JSON file named `classA.json`):
        {
            "image001": 4,
            "image002": {"channel": 5}
        }
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



def channel_custom_threshold(channel, threshold):
    """
    Apply custom thresholding to a single-channel image and extract YOLO-style 
    polygon annotations.

    This function performs binary thresholding followed by contour detection to 
    identify regions of interest in an image channel. It converts each contour 
    to a flattened polygon in YOLO-normalized coordinates (relative to the 
    target image size).

    Args:
        channel (np.ndarray): 2D NumPy array representing a single grayscale image channel.
        threshold (float): 
            - If < 1, interpreted as a percentile (e.g., 0.9 for the top 10% brightest pixels).
            - If ≥ 1, interpreted as an absolute pixel intensity threshold (0–255).

    Returns:
        tuple:
            - binary (np.ndarray): Binary thresholded image.
            - yolo_polygons (List[List[float]]): List of polygons where each polygon is a flattened list 
              [x1, y1, x2, y2, ..., xn, yn] in YOLO format (normalized to [0,1]).

    Notes:
        - The image size is taken from `HFinder_settings.get("target_size")`.
        - Only external contours are retained.
        - Contours with fewer than 3 points are discarded.
    """
    thresh_val = 0.9
    if threshold < 1:
        thresh_val = np.percentile(channel, threshold)
    else:
        # is a pixel value
        thresh_val = threshold

    w, h = HFinder_settings.get("target_size")
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



def channel_auto_threshold(channel):
    """
    Apply automatic thresholding to a single-channel image to extract binary 
    masks and YOLO-style polygons.

    This is a wrapper around `channel_custom_threshold`, using the default threshold
    value defined in settings (in percent).

    Args:
        channel (np.ndarray): 2D NumPy array representing a single image channel.

    Returns:
        tuple:
            - binary (np.ndarray): Binary image resulting from thresholding.
            - yolo_polygons (List[List[float]]): List of flattened polygons, each representing
              a detected object in YOLO-normalized coordinates.
    """
    auto_threshold = HFinder_settings.get("default_auto_threshold")
    return channel_custom_threshold(channel, auto_threshold)



def channel_custom_segment(json_path, ratio):
    """
    Load and normalize segmentation polygons from a COCO-style JSON annotation file.

    This function reads segmentation annotations from a JSON file and rescales 
    the coordinates to match the normalized YOLO format, according to a given 
    `ratio` and the target image size defined in settings.

    Args:
        json_path (str): Path to the JSON file containing COCO-style annotations.
        ratio (float): Scaling ratio between original image dimensions and the 
                       target size. Used to rescale polygon coordinates.

    Returns:
        List[List[float]]: A list of flattened polygons, where each polygon is represented 
        as a list of alternating x and y coordinates (normalized to [0,1]).

    Notes:
        - Only valid segmentation entries are processed.
        - Segments with missing or malformed data (e.g., odd number of coordinates) are skipped.
        - Coordinate normalization assumes the original segmentation is in absolute pixels
          and applies:  
              `x_new = x * ratio / width`,  
              `y_new = y * ratio / height`.
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
                HFinder_log.warn(f"Invalid segmentation in {json_path}, annotation id {ann.get('id')}")
                continue

            flat = [seg[i] * ratio / w if i % 2 == 0 else seg[i] * ratio / h for i in range(len(seg))]
            polygons.append(flat)

    return polygons



def prepare_class_inputs(folder_tree, base, channels, n, c, class_instructions, ratio):
    """
    Generate segmentation masks and polygon annotations per class, for each 
    frame or image channel. This function applies either:
      - custom pixel thresholding,
      - automatic thresholding,
      - or loads segmentation polygons from a JSON file (COCO-style),
    depending on the user-defined `class_instructions` for each semantic class.

    Args:
        folder_tree (dict): Dictionary representing the project folder structure.
                            Must include a "root" key pointing to the root directory.
        base (str): Base name for output files.
        channels (List[np.ndarray]): List of image channels, usually from a TIFF stack.
        n (int): Number of Z-stack frames (or images).
        c (int): Number of channels per frame.
        class_instructions (dict): Mapping of class names to processing instructions. Each value is either:
            - -1 (ignored class),
            - dict with "channel" and optional "threshold" (float or pixel value),
            - dict with "channel" and "segment" (path to JSON file).
        ratio (float): Scaling factor used to normalize coordinates when loading polygons from segmentation.

    Returns:
        defaultdict[list]: A mapping from frame indices to lists of (class_name, polygons),
                           where each polygon is a flattened list of normalized coordinates.

    Notes:
        - Thresholding is applied using OpenCV to generate binary masks and extract contours.
        - Resulting masks are saved as PNG files under `dataset/masks/`, named per class and frame.
        - If `"segment"` is used, polygons are extracted from a JSON file and scaled accordingly.
        - JSON segmentations are not supported for Z-stacks (`n > 1`); this raises a failure.
    """
    results = defaultdict(list)

    for class_name, instr in class_instructions.items():
        if instr == -1:
            continue
    
        ch = instr["channel"]

        if "threshold" in instr:
            for i in range(n):
                frame = i * c + ch
                binary, polygons = channel_custom_threshold(channels[frame], instr["threshold"])
                results[frame].append((class_name, polygons))
                name = f"dataset/masks/{base}_{class_name}_mask.png" if n == 1 \
                       else f"dataset/masks/{base}_frame{frame}_{class_name}_mask.png"
                binary_output = os.path.join(folder_tree["root"], name)
                plt.imsave(binary_output, binary, cmap='gray')

        elif "segment" in instr:
            if n > 1:
                HFinder_log.fail("Applying user segmentation to Z-stacks has not been implemented yet")
            json_path = os.path.join(HFinder_settings.get("tiff_dir"), instr["segment"])
            polygons = channel_custom_segment(json_path, ratio)
            results[ch].append((class_name, polygons))

        else:
            for i in range(n):
                frame = i * c + ch
                binary, polygons = channel_auto_threshold(channels[frame])
                results[frame].append((class_name, polygons))
                name = f"dataset/masks/{base}_{class_name}_mask.png" if n == 1 \
                       else f"dataset/masks/{base}_frame{frame}_{class_name}_mask.png"
                binary_output = os.path.join(folder_tree["root"], name)
                plt.imsave(binary_output, binary, cmap='gray')

    return results



def is_channel_first(img):
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
 
    n = 1
    if img.ndim == 4:
        n, c, h, w = img.shape
    else:
        c, h, w = img.shape
    resized = np.empty((n * c, *target_size), dtype=img.dtype)
    for n_i in range(n):
        for c_i in range(c):
            index = n_i * c + c_i
            frame = img[c_i] if n == 1 else img[n_i][c_i]
            resized[index] = cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    ratios = tuple(x / w for x in target_size)
    assert(ratios[0] == ratios[1]) # TODO: Remove this?
    return {i + 1: resized[i] for i in range(n * c)}, ratios[0], (n, c)



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



def generate_contours(folder_tree, base, polygons_per_channel, channels, class_ids):
    root = folder_tree["root"]
    contour_dir = os.path.join(root, "dataset", "contours")

    for ch_name, polygons in polygons_per_channel.items():
        img = channels[ch_name]
        h, w = img.shape
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for class_name, poly in polygons:
            if class_name not in class_ids:
                continue
            class_id = class_ids[class_name]
            color = HFinder_palette.get_color(class_id)

            for poly in poly:  # liste de polygones
                # poly est une liste plate : [x1, y1, x2, y2, ..., xn, yn]
                if len(poly) < 6:
                    continue  # ignore les artefacts trop petits (moins de 3 points)

                pts = np.array(
                    [(int(poly[i] * w), int(poly[i + 1] * h)) for i in range(0, len(poly), 2)],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)

        out_path = os.path.join(contour_dir, f"{base}_{ch_name}_contours.png")
        cv2.imwrite(out_path, overlay)



def generate_dataset(folder_tree, base, n, c, split_flag, channels, polygons_per_channel):
    def power_set(channels):
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
    def compose_hue_fusion(channels, selected_channels, palette, noise_channels=None):
        """
        Compose an RGB image by blending each selected channel with a random hue.
        """
        h, w = next(iter(channels.values())).shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for ch in selected_channels:
            frame = channels[ch]
            hue = HFinder_palette.get_color(ch, palette=palette)[0] #np.random.rand()
            colored = colorize_with_hue(frame, hue) 
            rgb += colored  # additive mixing

        # Ajout de bruit visuel contrôlé
        if noise_channels:
            for ch in noise_channels:
                frame = channels[ch]
                hue = HFinder_palette.get_color(ch, palette=palette)[0] #np.random.rand()
                noise = colorize_with_hue(frame, hue)
                rgb += noise  # Opacité réduite du bruit

        # Clip to [0, 1] in case of saturation, then scale to [0, 255]
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)

    def convert_polygon_to_yolo_segmentation(polygon):
        return [coord for point in polygon for coord in point]

    def make_filename(base, combo):
        return f"{os.path.splitext(base)[0]}_" + "_".join(map(str, combo)) + ".jpg"

    def save_image_as_jpg(img_rgb, path):
        Image.fromarray(img_rgb).save(path, "JPEG")

    def save_yolo_segmentation_label(file_path, polygons, class_ids, img_w, img_h):
        with open(file_path, "w") as f:
            for class_name, poly in polygons:
                if class_name not in class_ids:
                    continue
                class_id = class_ids[class_name]
                poly = convert_polygon_to_yolo_segmentation(poly)
                line = [str(class_id)] + [f"{x:.6f}" for x in poly]
                f.write(" ".join(line) + "\n")

    # Détermination du sous-dossier (train/val)
    split = "train" if split_flag else "val"
    img_dir = os.path.join(folder_tree["root"], "dataset", "images", split)
    lbl_dir = os.path.join(folder_tree["root"], "dataset", "labels", split)

    # Chargement des identifiants de classes
    class_ids = load_class_definitions()  # suppose définie ailleurs
    target_size = HFinder_settings.get("target_size")
    img_h, img_w = target_size

    annotated_channels = {ch for ch, polys in polygons_per_channel.items() if polys}
    all_channels = set(channels.keys())

    #print(f"polygons_per_channel.keys() = {polygons_per_channel.keys()}, list(annotated_channels) = {list(annotated_channels)}")
    for combo in power_set(annotated_channels):
        filename = make_filename(base, combo)
        noise_candidates = list(all_channels - set(combo))
        if n > 1:
        # Prendre le canal avec l'indice minimum pour déduire le niveau
            ref_ch = min(combo)
            series_index = (ref_ch - 1) // c
            allowed_noise = [series_index * c + i + 1 for i in range(c)]
            noise_candidates = list(set(noise_candidates) & set(allowed_noise))
            # Échantillonnage aléatoire des canaux de bruit
            num_noise = np.random.randint(0, len(noise_candidates) + 1)
            noise_channels = random.sample(noise_candidates, num_noise) if num_noise > 0 else []
        else:
            noise_channels = random.sample(noise_candidates, k=random.randint(1, len(noise_candidates)))
        #print(f"Image {base}, combo = {combo}, noise_channels = {noise_channels}, noise_candidates = {noise_candidates}")
        special_palette = HFinder_palette.get_random_palette(colorspace="HSV", hash_data=filename)
        img_rgb = compose_hue_fusion(channels, combo, special_palette, noise_channels=noise_channels)
        img_path = os.path.join(img_dir, filename)
        save_image_as_jpg(img_rgb, img_path)

        # Annotations combinées
        annotations = list(chain.from_iterable(polygons_per_channel.get(ch, []) for ch in combo))
        if annotations:
            label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
            save_yolo_segmentation_label(label_path, annotations, class_ids, img_w, img_h)



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
        HFinder_log.info(f"Processing {img_name}")
        base = os.path.splitext(img_name)[0]
        if img_name not in class_instructions:
            HFinder_log.warn(f"Skipping file {img_name} - no annotations")
            continue  # skip images not listed

        # Ensure this is a proper TIFF file.
        # FIXME: handle z-stacks and time series.
        img = tifffile.imread(img_path)
        if not is_channel_first(img):
            HFinder_log.warn(f"Skipping file {img_name}, wrong shape {img.shape}")
            continue

        # Resize image and split channels.
        channels, ratio, (n, c) = resize_multichannel_image(img)   
        
        # Retrieve polygons for each class and generate dataset.  
        polygons_per_channel = prepare_class_inputs(folder_tree, base, channels, n, c, class_instructions[img_name], ratio)  
        generate_contours(folder_tree, base, polygons_per_channel, channels, class_ids)     
        generate_dataset(folder_tree, base, n, c, split_table[img_path], channels, polygons_per_channel)

