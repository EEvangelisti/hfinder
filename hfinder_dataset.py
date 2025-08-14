"""
High-level dataset preparation for HFinder.

Overview
--------
- Load per-image class instructions and resolve per-channel operations.
- Threshold or segment channels to produce masks and polygons.
- Convert masks/contours to YOLOv8-compatible polygon labels.
- Compose RGB training images via hue-based fusion of channels.
- Save images/labels to the expected YOLO directory layout.
- Split images into train/val subsets.
- Optionally compute and export MIP (Max Intensity Projection) datasets.

Public API
----------
- prepare_class_inputs(channels, n, c, ratio)
    Build per-frame annotations (polygons) from thresholds or JSON segmentations.
- generate_contours(base, polygons_per_channel, channels, class_ids)
    Render filled/outlined polygons as visual overlays for QA.
- generate_dataset(base, n, c, channels, polygons_per_channel)
    Create RGB composites and YOLO segmentation labels for training.
- split_train_val()
    Move a fraction of training images (and labels) to validation.
- max_intensity_projection_multichannel(img_name, base, stack, polygons_per_channel, class_ids, n, c, ratio)
    Produce a MIP per channel, aggregate polygons across Z, export overlays and dataset.
- generate_training_dataset()
    Orchestrate the full dataset generation flow from input TIFFs.

Notes
-----
- Channel indexing is 1-based throughout (consistent with upstream modules).
- Hue fusion uses deterministic palette rotation when hashing the filename,
  enabling reproducible visuals.
- JSON polygon application to Z-stacks is not implemented (explicit failure).
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
    frame or image channel, based on class-specific directives.

    For each class (from HFinder_ImageInfo):
      - If a custom threshold is defined: threshold → mask → contours → polygons.
      - Else if a manual segmentation JSON is provided: load polygons (scaled).
      - Else: apply automatic thresholding → mask → contours → polygons.

    :param channels: Dict mapping 1-based channel/frame index → 2D image array.
    :type channels: dict[int, np.ndarray]
    :param n: Number of Z frames (1 if single plane).
    :type n: int
    :param c: Channels per Z frame.
    :type c: int
    :param ratio: Resize ratio (target_size / original_width) for polygon scaling.
    :type ratio: float
    :return: Map frame index → list of (class_name, [flat_polygons...]).
    :rtype: collections.defaultdict[list]

    :notes:
        - Masks are saved under dataset/masks/, per class/frame.
        - JSON segmentations on Z-stacks (n > 1) fail with an explicit message.
        - Flat polygons are lists like [x1, y1, ..., xn, yn], normalized to [0, 1].
    """
    results = defaultdict(list)
    masks_dir = HFinder_folders.get_masks_dir()
    base = HFinder_settings.get("current_image.base")

    for cls in HFinder_ImageInfo.get_classes():
    
        # Per-class directives
        ch = HFinder_ImageInfo.get_channel(cls)
        threshold = HFinder_ImageInfo.get_threshold(cls)
        poly_json = HFinder_ImageInfo.get_manual_segmentation(cls)

        if threshold is not None:
            # Custom (fixed) thresholding across a frame range
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
            # Load user-provided segmentation polygons (single-plane only)
            if n > 1:
                HFinder_log.fail("Applying user segmentation to Z-stacks has not been implemented yet")
            json_path = os.path.join(HFinder_settings.get("tiff_dir"), poly_json)
            polygons = HFinder_segmentation.channel_custom_segment(json_path, ratio)
            results[ch].append((cls, polygons))

        else:
            # Automatic thresholding as a fallback
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
    """
    Draw filled/outlined polygons over grayscale channels and save overlays.

    :param base: Base name used for output files.
    :type base: str
    :param polygons_per_channel: Map channel → [(class_name, [polys...]), ...].
    :type polygons_per_channel: dict[int, list[tuple[str, list[list[float]]]]]
    :param channels: Dict channel → 2D grayscale image.
    :type channels: dict[int, np.ndarray]
    :param class_ids: Class name → integer ID mapping (ignored here except for filtering).
    :type class_ids: dict[str, int]
    :rtype: None
    """
    contours_dir = HFinder_folders.get_contours_dir()

    for ch_name, polygons in polygons_per_channel.items():
        channel = channels[ch_name]
        h, w = channel.shape
        overlay = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)

        for class_name, poly in polygons:
            if class_name not in class_ids:
                continue

            for poly in poly:
                # Flat [x1, y1, ..., xn, yn]; ignore degenerate polygons
                if len(poly) < 6:
                    continue  # ignore artifacts (tiny polygons)

                pts = np.array(
                    [(int(poly[i] * w), int(poly[i + 1] * h)) for i in range(0, len(poly), 2)],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                # Choose color (fixed for publication, random for exploration)
                if HFinder_settings.get("publication"):
                    color = (255, 0, 255)
                else:
                    color = tuple(random.randint(10, 255) for _ in range(3))

                overlay_copy = overlay.copy()
                # Fill on a copy
                cv2.fillPoly(overlay_copy, [pts], color)
                alpha = 0.3
                overlay = cv2.addWeighted(overlay_copy, alpha, overlay, 1 - alpha, 0)
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)
                if not HFinder_settings.get("publication"):
                    white = (255, 255, 255)
                    cv2.putText(
                        overlay, class_name, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1
                    )

        out_path = os.path.join(contours_dir, f"{base}_{ch_name}_contours.png")
        cv2.imwrite(out_path, overlay)



def generate_dataset(base, n, c, channels, polygons_per_channel):
    """
    Compose RGB training images via hue fusion and export YOLO segmentation labels.

    For each combination of annotated channels (per Z-frame if applicable):
      - Sample optional "noise" channels (never annotated ones; same Z-frame for n>1).
      - Compose a hue-fused RGB image (selected+noise).
      - Save image as JPEG under dataset/images/train/.
      - Export YOLO polygon labels if any annotated polygons exist for the combo.

    :param base: Base filename for images/labels.
    :type base: str
    :param n: Number of Z frames (1 if single plane).
    :type n: int
    :param c: Channels per Z frame.
    :type c: int
    :param channels: Dict channel → 2D grayscale image.
    :type channels: dict[int, np.ndarray]
    :param polygons_per_channel: Dict channel → [(class_name, [polys...]), ...].
    :type polygons_per_channel: dict[int, list[tuple[str, list[list[float]]]]]
    :rtype: None
    """
    img_dir = HFinder_folders.get_image_train_dir()
    lbl_dir = HFinder_folders.get_label_train_dir()

    class_ids = HFinder_utils.load_class_definitions()

    annotated_channels = {ch for ch, polys in polygons_per_channel.items() if polys}
    all_channels = set(channels.keys())

    if HFinder_settings.get("mode") == "debug":
        print(f"polygons_per_channel.keys() = {polygons_per_channel.keys()}, \
                list(annotated_channels) = {list(annotated_channels)}")

    # Iterate over valid channel combinations (per Z-frame if n>1)
    for combo in HFinder_utils.power_set(annotated_channels, n, c):
        filename = f"{os.path.splitext(base)[0]}_" + "_".join(map(str, combo)) + ".jpg"
        
        # Never use annotated channels as noise (even if not in the current combo)
        noise_candidates = list(all_channels - set(annotated_channels) - set(combo))

        if n > 1:
            # Restrict noise to the Z-frame of the combo reference channel
            ref_ch = min(combo)
            series_index = (ref_ch - 1) // c
            allowed_noise = [series_index * c + i + 1 for i in range(c)]
            noise_candidates = list(set(noise_candidates) & set(allowed_noise))

        # Sample 0..len(noise_candidates) noise channels
        num_noise = np.random.randint(0, len(noise_candidates) + 1)
        noise_channels = random.sample(noise_candidates, num_noise) if num_noise > 0 else []

        if HFinder_settings.get("mode") == "debug":
            print(f"Image {base}, combo = {combo}, \
                    noise_channels = {noise_channels}, \
                    noise_candidates = {noise_candidates}")

        # Compose hue-fused RGB and save
        palette = HFinder_palette.get_random_palette(hash_data=filename)
        img_rgb = HFinder_ImageOps.compose_hue_fusion(
            channels, combo, palette, noise_channels=noise_channels
        )
        img_path = os.path.join(img_dir, filename)
        Image.fromarray(img_rgb).save(img_path, "JPEG")

        # Flatten annotations for the selected channels and write YOLO labels.
        annotations = list(chain.from_iterable(polygons_per_channel.get(ch, []) for ch in combo))
        if annotations:
            label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
            HFinder_utils.save_yolo_segmentation_label(label_path, annotations, class_ids)



def split_train_val():
    """
    Split images/labels into training and validation subsets.

    Moves a fraction of images (and their labels) from train → val according
    to `validation_frac` in settings.

    :rtype: None
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
    """
    Build a Max Intensity Projection (MIP) across Z and export a mini-dataset.

    Steps per channel index 0..c-1:
      - Compute the MIP across n frames → resize to target size.
      - Aggregate polygons across all Z-slices for each class.
      - Fill a global mask per class and convert to YOLO polygons.
      - Save masks and overlays; export a small MIP-based dataset.

    :param img_name: Original image filename (for bookkeeping).
    :type img_name: str
    :param base: Base name for outputs (suffix "_MIP" is added).
    :type base: str
    :param stack: Original image stack (ndarray with Z).
    :type stack: np.ndarray
    :param polygons_per_channel: Dict channel_idx → [(class_name, [polys...]), ...].
    :type polygons_per_channel: dict[int, list[tuple[str, list[list[float]]]]]
    :param class_ids: Mapping class name → ID.
    :type class_ids: dict[str, int]
    :param n: Number of Z frames.
    :type n: int
    :param c: Channels per frame.
    :type c: int
    :param ratio: Resize factor used elsewhere (kept for parity).
    :type ratio: float
    :rtype: None

    :notes:
        - This routine assumes square targets (size × size).
        - It aggregates instance polygons across Z before re-vectorizing.
    """
    mip = np.max(stack, axis=0)   # shape: (C, H, W)
    stacked_channels = [
        cv2.resize(mip[ch], (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        for ch in range(c)
    ]
    size = HFinder_settings.get("size")
    assert (stacked_channels[0].shape == (size, size))

    masks_dir = HFinder_folders.get_masks_dir()
    contours_dir = HFinder_folders.get_contours_dir()

    # For each channel of the MIP, merge polygons from all Z-slices
    polygons_per_stacked_channel = defaultdict(list)
    for ch in range(c):
        # 1-based indices of frames for this channel across Z: 1, 1+c, 1+2c, ...
        indices = [j + 1 for j in range(ch, n * c, c)]  # 1, 1+c, 1+2c, ...
        
        # Collect per-slice polygons for the channel: [(label, [polys...]), ...]
        polygons_subset = [polygons_per_channel.get(idx, []) for idx in indices]

        # Build a fused mask per class from all Z-slice polygons
        allowed_items = [(x, y) for x, y in class_ids.items() if HFinder_ImageInfo.allows_MIP_generation(x)]
        for class_name, class_id in allowed_items:
            # Accumulate polygons (pixel coords) for this class across slices
            all_polys_px = []
            for per_slice in polygons_subset:
                for label, polys_list in per_slice:
                    if label != class_name:
                        continue
                    # polys_list est une liste de polygones plats
                    for flat in polys_list:
                        if not flat: 
                            continue
                        # Convert normalized flat polygon to pixel coordinates
                        pts = HFinder_geometry.flat_to_pts_xy(flat)   # (N, 2)
                        if pts.shape[0] >= 3:
                            all_polys_px.append(pts)

            # If nothing accumulated, skip this class/channel
            if not all_polys_px:
                continue

            # Fused binary mask for the class on the MIP channel
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, all_polys_px, 255)

            # Persist the fused class mask (for QA or reuse)
            mask_path = os.path.join(masks_dir, f"{base}_MIP_{class_name}_mask.png")
            cv2.imwrite(mask_path, mask)

            # Extract refined contours and convert to YOLO polygons
            final_contours, _ = HFinder_segmentation.find_fine_contours(
                mask, scale=3, canny=True, eps=0.3, min_perimeter=25
            )
            yolo_polygons = HFinder_geometry.contours_to_yolo_polygons(final_contours)
 
            ch_key = ch + 1   # 1-based
            polygons_per_stacked_channel[ch_key].append((class_name, yolo_polygons))

    # Compose overlays and export a small MIP dataset
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
    """
    End-to-end dataset generation from a folder of TIFFs.

    Steps:
      - Discover input images under `tiff_dir`.
      - Initialize class instructions and write dataset YAML.
      - For each image:
          * Validate shape; resize/stack channels.
          * Create class-specific masks/annotations.
          * Save overlays and fused RGB training examples (+ labels).
          * Optionally export MIP-based dataset if multiple Z-frames.
      - Finally split train/val per `validation_frac`.

    :rtype: None
    """
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

        # Resize channels to the configured size; get ratio and (n, c)
        channels, ratio, (n, c) = HFinder_ImageOps.resize_multichannel_image(img)   
        polygons_per_channel = prepare_class_inputs(channels, n, c, ratio)
        
        # QA overlays then dataset generation
        generate_contours(base, polygons_per_channel, channels, class_ids)     
        generate_dataset(base, n, c, channels, polygons_per_channel)

        # Optional MIP export when multiple Z-slices exist
        if n > 1:
            max_intensity_projection_multichannel(
                img_name, base, img, polygons_per_channel,
                class_ids, n, c, ratio
            )

    # Split train/val at the end so both single-plane and MIP outputs are included
    split_train_val()
