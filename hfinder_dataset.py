"""
HFinder_dataset

This module provides high-level functions for preparing and managing training 
datasets for the HFinder pipeline. It includes utilities to:

- Load and interpret image-class mappings and annotation instructions.
- Perform thresholding and segmentation on individual image channels.
- Convert image masks or manual annotations into polygon representations.
- Generate datasets in YOLOv8 format, including image-label pairs.
- Split datasets into training and validation subsets.
- Perform max intensity projections of multichannel stacks.

These functions orchestrate the transformation of raw microscopy data into a 
machine-learning-ready format for semantic segmentation tasks.

Typical use cases include:
    - Preparing training data from raw image stacks and JSON annotations.
    - Exporting YOLO-compatible segmentation labels.
    - Automating dataset generation for model training.

"""

import os
import cv2
import json
import random
import shutil
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from itertools import chain
from collections import defaultdict
import hfinder_log as HFinder_log
import hfinder_utils as HFinder_utils
import hfinder_folders as HFinder_folders
import hfinder_palette as HFinder_palette
import hfinder_imageinfo as HFinder_ImageInfo
import hfinder_settings as HFinder_settings
import hfinder_imageops as HFinder_ImageOps
import hfinder_geometry as HFinder_geometry
import hfinder_segmentation as HFinder_segmentation





def prepare_class_inputs(channels, n, c, ratio):
    """
    Generate segmentation masks and polygon annotations per class, for each 
    frame or image channel. This function applies either:
      - custom pixel thresholding,
      - automatic thresholding,
      - or loads segmentation polygons from a JSON file (COCO-style),
    depending on the user-defined `class_instructions` for each semantic class.

    Args:
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
    masks_dir = HFinder_folders.get_masks_dir()
    base = HFinder_settings.get("current_image.base")

    for cls in HFinder_ImageInfo.get_classes():
    
        ch = HFinder_ImageInfo.get_channel(cls)
        threshold = HFinder_ImageInfo.get_threshold(cls)
        poly_json = HFinder_ImageInfo.get_manual_segmentation(cls)

        if threshold is not None:
            from_frame = HFinder_ImageInfo.from_frame(cls, default=0)
            to_frame = HFinder_ImageInfo.to_frame(cls, default=n)
            for i in range(from_frame // c, to_frame // c + 1):
                frame = i * c + ch
                binary, polygons = HFinder_segmentation.channel_custom_threshold(channels[frame], threshold)
                results[frame].append((cls, polygons))
                name = f"{base}_{cls}_mask.png" if n == 1 \
                       else f"{base}_frame{frame}_{cls}_mask.png"
                binary_output = os.path.join(masks_dir, name)
                plt.imsave(binary_output, binary, cmap='gray')

        elif poly_json is not None:
            if n > 1:
                HFinder_log.fail("Applying user segmentation to Z-stacks has not been implemented yet")
            json_path = os.path.join(HFinder_settings.get("tiff_dir"), poly_json)
            polygons = HFinder_segmentation.channel_custom_segment(json_path, ratio)
            results[ch].append((cls, polygons))

        else:
            from_frame = HFinder_ImageInfo.from_frame(cls, default=0)
            to_frame = HFinder_ImageInfo.to_frame(cls, default=n)
            for i in range(from_frame // c, to_frame // c + 1):
                frame = i * c + ch
                binary, polygons = HFinder_segmentation.channel_auto_threshold(channels[frame])
                results[frame].append((cls, polygons))
                name = f"{base}_{cls}_mask.png" if n == 1 \
                       else f"{base}_frame{frame}_{cls}_mask.png"
                binary_output = os.path.join(masks_dir, name)
                plt.imsave(binary_output, binary, cmap='gray')

    return results



def generate_contours(base, polygons_per_channel, channels, class_ids):
    contours_dir = HFinder_folders.get_contours_dir()

    for ch_name, polygons in polygons_per_channel.items():
        channel = channels[ch_name]
        h, w = channel.shape
        overlay = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)

        for class_name, poly in polygons:
            if class_name not in class_ids:
                continue
            class_id = class_ids[class_name]

            for poly in poly:
                # poly is a flat list: [x1, y1, x2, y2, ..., xn, yn]
                if len(poly) < 6:
                    continue  # ignore artifacts (tiny polygons)

                pts = np.array(
                    [(int(poly[i] * w), int(poly[i + 1] * h)) for i in range(0, len(poly), 2)],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                color = tuple(random.randint(10, 255) for _ in range(3))
                overlay_copy = overlay.copy()
                # Fill polygon on the overlay copy
                cv2.fillPoly(overlay_copy, [pts], color)
                # Blend filled polygon into original overlay with alpha=0.3
                alpha = 0.3
                overlay = cv2.addWeighted(overlay_copy, alpha, overlay, 1 - alpha, 0)
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)
                cv2.putText(overlay, class_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = os.path.join(contours_dir, f"{base}_{ch_name}_contours.png")
        cv2.imwrite(out_path, overlay)



def generate_dataset(base, n, c, channels, polygons_per_channel):
    """
    Generates training data for YOLO-based segmentation by fusing image channels
    into RGB composites and assigning segmentation masks based on annotated 
    polygon data.

    Parameters:
        base (str): Base filename used to construct output image and label names.
        n (int): Number of spatial/Z positions (frames) to combine per training example.
        c (int): Number of channels per spatial position.
        channels (dict[int → np.ndarray]): Dictionary mapping channel indices 
        to grayscale image arrays.
        polygons_per_channel (dict[int → list[tuple]]): Maps channel indices to 
        lists of (class_name, polygon) tuples representing instance segmentations.

    Functionality:
        Iterates over all valid combinations (combo) of n spatial positions 
        and c channels among annotated channels. For each combination:
            - Excludes all annotated channels from the pool of possible "noise" 
              channels (non-annotated distractors).
            - Restricts possible noise channels to the same Z/t position as the 
              combo if n > 1.
            - Randomly adds zero or more noise channels to the input image.
            - Composes a pseudo-RGB image using a hue-fusion strategy based on 
              the combo and noise channels.
            - Saves the resulting image as a .jpg file in the YOLO 
              "train/images" folder.
            - If polygons are available for any of the combo channels, it
              generates and saves a YOLO-format .txt segmentation label.

    Strict separation between annotated and noise channels ensures no meaningful
    biological signal appears without its corresponding annotation — a crucial 
    constraint to prevent inconsistencies during training. The approach supports
    multiple Z/t frames per sample (via n > 1) and random noise injection for
    data augmentation. Uses deterministic random color palettes (via hashing) 
    for consistent hue fusion in debug or reproducibility scenarios.

    Outputs:
        JPEG images and corresponding YOLO segmentation labels, saved under:
        dataset/images/train/
        dataset/labels/train/
    """
    train_dir = os.path.join("dataset", "{}", "train")
    img_dir = HFinder_folders.get_image_train_dir()
    lbl_dir = HFinder_folders.get_label_train_dir()

    class_ids = HFinder_utils.load_class_definitions()
    target_size = HFinder_settings.get("target_size")

    annotated_channels = {ch for ch, polys in polygons_per_channel.items() if polys}
    all_channels = set(channels.keys())

    if HFinder_settings.get("mode") == "debug":
        print(f"polygons_per_channel.keys() = {polygons_per_channel.keys()}, \
                list(annotated_channels) = {list(annotated_channels)}")

    for combo in HFinder_utils.power_set(annotated_channels, n, c):
        filename = f"{os.path.splitext(base)[0]}_" + "_".join(map(str, combo)) + ".jpg"
        
        # Any channel containing annotations must never be used as background 
        # noise, even when not selected for the current input combination. This 
        # avoids showing biologically meaningful signal without its associated 
        # mask, which would create inconsistency during training.
        noise_candidates = list(all_channels - set(annotated_channels) - set(combo))

        if n > 1:
            # We can only select channels at the same Z/t level than combo.
            ref_ch = min(combo)
            series_index = (ref_ch - 1) // c
            allowed_noise = [series_index * c + i + 1 for i in range(c)]
            noise_candidates = list(set(noise_candidates) & set(allowed_noise))

        num_noise = np.random.randint(0, len(noise_candidates) + 1)
        noise_channels = random.sample(noise_candidates, num_noise) if num_noise > 0 else []

        if HFinder_settings.get("mode") == "debug":
            print(f"Image {base}, combo = {combo}, \
                    noise_channels = {noise_channels}, \
                    noise_candidates = {noise_candidates}")

        palette = HFinder_palette.get_random_palette(hash_data=filename)
        img_rgb = HFinder_ImageOps.compose_hue_fusion(channels, combo, palette, noise_channels=noise_channels)
        img_path = os.path.join(img_dir, filename)
        Image.fromarray(img_rgb).save(img_path, "JPEG")

        annotations = list(chain.from_iterable(polygons_per_channel.get(ch, []) for ch in combo))
        if annotations:
            label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
            HFinder_utils.save_yolo_segmentation_label(label_path, annotations, class_ids)



def split_train_val():
    """
    Splits the dataset into training and validation sets. This function moves 
    (1 - percent) of the images from the training directory to the validation 
    directory, along with their corresponding label files.
    """
    img_dir = HFinder_folders.get_image_train_dir()
    lbl_dir = HFinder_folders.get_label_train_dir()
    img_val_dir = HFinder_folders.get_image_val_dir()
    lbl_val_dir = HFinder_folders.get_label_val_dir()

    image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    random.shuffle(image_paths)

    percent = HFinder_settings.get("validation_frac")
    val_count = int(len(image_paths) * percent)
    val_images = image_paths[:val_count]

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"

        src_lbl_path = os.path.join(lbl_dir, label_name)
        dst_img_path = os.path.join(img_val_dir, img_name)
        dst_lbl_path = os.path.join(lbl_val_dir, label_name)

        shutil.move(img_path, dst_img_path)
        shutil.move(src_lbl_path, dst_lbl_path)




def max_intensity_projection_multichannel(img_name, base, stack, polygons_per_channel, class_ids, n, c, ratio):
    mip = np.max(stack, axis=0)
    stacked_channels = [
        cv2.resize(mip[ch], (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        for ch in range(c)
    ]
    w, h = HFinder_settings.get("target_size")
    assert (stacked_channels[0].shape == (w, h))

    masks_dir = HFinder_folders.get_masks_dir()
    contours_dir = HFinder_folders.get_contours_dir()

    # Pour chaque canal de la MIP, on fusionne tous les polygones de ce canal à travers Z
    polygons_per_stacked_channel = defaultdict(list)
    for ch in range(c):
        # indices (1-based) des frames correspondant à ce canal à travers les n plans
        indices = [j + 1 for j in range(ch, n * c, c)]  # 1, 1+c, 1+2c, ...
        # polygons_per_channel[idx] est une liste de tuples (class_name, [poly1, poly2, ...])
        polygons_subset = [polygons_per_channel.get(idx, []) for idx in indices]

        # Pour chaque classe, construire un masque fusionné
        allowed_items = [(x, y) for x, y in class_ids.items() if HFinder_ImageInfo.allows_MIP_generation(x)]
        for class_name, class_id in allowed_items:
            # Collecte des polygones plats pour cette classe sur toutes les slices
            all_polys_px = []
            for per_slice in polygons_subset:
                for label, polys_list in per_slice:
                    if label != class_name:
                        continue
                    # polys_list est une liste de polygones plats
                    for flat in polys_list:
                        if not flat: 
                            continue
                        pts = HFinder_geometry.flat_to_pts_xy(flat, w, h) # (N,2)
                        # OpenCV accepte (N,2) ou (N,1,2); (N,2) suffit pour fillPoly
                        if pts.shape[0] >= 3:
                            all_polys_px.append(pts)

            # Si rien pour cette classe → passer
            if not all_polys_px:
                continue

            # Masque fusionné
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, all_polys_px, 255)

            # Sauvegarde du masque
            mask_path = os.path.join(masks_dir, f"{base}_MIP_{class_name}_mask.png")
            cv2.imwrite(mask_path, mask)

            # TODO: Check whether it works
            final_contours, _ = HFinder_segmentation.find_fine_contours(mask, scale=3, canny=True, eps=0.3, min_perimeter=25)
            yolo_polygons = HFinder_geometry.contours_to_yolo_polygons(final_contours)
 
            ch_key = ch + 1
            polygons_per_stacked_channel[ch_key].append((class_name, yolo_polygons))

    stacked_channels_dict = {i+1: stacked_channels[i] for i in range(c)}
    generate_contours(
        base + "_MIP",
        polygons_per_stacked_channel, 
        stacked_channels_dict,
        class_ids
    )  

    generate_dataset(
        base + "_MIP",
        n=1, c=c,
        channels=stacked_channels_dict,
        polygons_per_channel=polygons_per_stacked_channel
    )



def generate_training_dataset():
    data_dir = HFinder_settings.get("tiff_dir")
    image_paths = sorted(glob(os.path.join(data_dir, "*.tif")))
    
    class_ids = HFinder_utils.load_class_definitions()
    HFinder_utils.write_yolo_yaml(class_ids)
    HFinder_ImageInfo.initialize()

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        HFinder_ImageInfo.set_current_image(img_name)
        base = os.path.splitext(img_name)[0]
        HFinder_settings.set("current_image.base", base)

        if not HFinder_ImageInfo.image_has_instructions():
            HFinder_log.warn(f"Skipping file {img_name} - no annotations")
            continue

        img = tifffile.imread(img_path)
        if not HFinder_ImageOps.is_valid_image_format(img):
            HFinder_log.warn(f"Skipping file {img_name}, wrong shape {img.shape}")
            continue

        channels, ratio, (n, c) = HFinder_ImageOps.resize_multichannel_image(img)   
        polygons_per_channel = prepare_class_inputs(channels, n, c, ratio)
        generate_contours(base, polygons_per_channel, channels, class_ids)     
        generate_dataset(base, n, c, channels, polygons_per_channel)

        if n > 1:
            max_intensity_projection_multichannel(img_name, base, img, polygons_per_channel, class_ids, n, c, ratio)

    split_train_val()
