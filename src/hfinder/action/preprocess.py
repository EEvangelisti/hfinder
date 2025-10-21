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
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from itertools import chain
from collections import defaultdict
from hfinder.core import log as HF_log
from hfinder.core import utils as HF_utils
from hfinder.core import palette as HF_palette
from hfinder.core import geometry as HF_geometry
from hfinder.image import processing as HF_ImageOps
from hfinder.image import segmentation as HF_segmentation
from hfinder.session import current as HF_ImageInfo
from hfinder.session import folders as HF_folders
from hfinder.session import settings as HF_settings



def prepare_class_inputs(channels, n, c, ratio):
    """
    Generate segmentation masks and polygon annotations per class, for each
    frame or image channel, based on class-specific directives.

    For each class (from HF_ImageInfo):
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
    masks_dir = HF_folders.get_masks_dir()
    base = HF_settings.get("current_image.base")

    for cls in HF_ImageInfo.get_classes():
    
        HF_ImageInfo.set_current_class(cls)
        # Per-class directives
        ch = HF_ImageInfo.get_channel()
        threshold = HF_ImageInfo.get_threshold()
        poly_json = HF_ImageInfo.get_manual_segmentation()

        if threshold is not None:
            # Custom (fixed) thresholding across a frame range
            from_frame = HF_ImageInfo.from_frame(default=0)
            to_frame = HF_ImageInfo.to_frame(default=n)
            for i in range(from_frame // c, to_frame // c + 1):
                frame = i * c + ch
                binary, polygons = HF_segmentation.channel_custom_threshold(channels[frame], threshold)
                results[frame].append((cls, polygons))
                name = f"{base}_{cls}_mask.png" if n == 1 \
                       else f"{base}_frame{frame}_{cls}_mask.png"
                binary_output = os.path.join(masks_dir, name)
                plt.imsave(binary_output, binary, cmap='gray')

        elif poly_json is not None:
            # Load user-provided segmentation polygons (single-plane only)
            if n > 1:
                HF_log.fail(f"File '{base}.tif' - applying user segmentation to Z-stacks has not been implemented yet")
            json_path = os.path.join(HF_settings.get("tiff_dir"), poly_json)
            polygons = HF_segmentation.channel_custom_segment(json_path, ratio)
            results[ch].append((cls, polygons))

        else:
            # Automatic thresholding as a fallback
            from_frame = HF_ImageInfo.from_frame(default=0)
            to_frame = HF_ImageInfo.to_frame(default=n)
            for i in range(from_frame // c, to_frame // c + 1):
                frame = i * c + ch
                binary, polygons = HF_segmentation.channel_auto_threshold(channels[frame])
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
    contours_dir = HF_folders.get_contours_dir()

    for ch_name, polygons in polygons_per_channel.items():
        channel = channels[ch_name]
        h, w = channel.shape
        overlay = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)

        unique_classes = sorted({cn for cn, _ in polygons})
        multi_class = len(unique_classes) > 1
        label_positions = {}

        if multi_class:
            # Stable palette if class set is the same (order-independent)
            hsv_palette = HF_palette.get_random_palette(hash_data="|".join(unique_classes))

            def hsv_to_bgr(h, s, v):
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                return (int(r * 255), int(g * 255), int(b * 255))  # OpenCV uses BGR

            # Map each class to a palette color (cycle if more classes than palette size)
            class_colors = {
                cn: hsv_to_bgr(*hsv_palette[i % len(hsv_palette)])
                for i, cn in enumerate(unique_classes)
            }
        else:
            class_colors = None  # fall back to original behavior


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
                if multi_class:
                    color = class_colors[class_name]
                else:
                    if HF_settings.get("publication"):
                        color = (255, 0, 255)
                    else:
                        color = tuple(random.randint(10, 255) for _ in range(3))

                overlay_copy = overlay.copy()
                # Fill on a copy
                cv2.fillPoly(overlay_copy, [pts], color)
                alpha = 0.3
                overlay = cv2.addWeighted(overlay_copy, alpha, overlay, 1 - alpha, 0)
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=1)
                if multi_class or not HF_settings.get("publication"):
                    if multi_class:
                        # Stack class names: one line per class, fixed horizontal position
                        if class_name not in label_positions:
                            # Assign next available vertical offset
                            y_offset = 20 + 20 * len(label_positions)
                            label_positions[class_name] = y_offset

                        y_pos = label_positions[class_name]
                        cv2.putText(
                            overlay, class_name, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                        )
                    else:
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
    img_dir = HF_folders.get_image_train_dir()
    lbl_dir = HF_folders.get_label_train_dir()

    class_ids = HF_settings.load_class_definitions()

    annotated_channels = {ch for ch, polys in polygons_per_channel.items() if polys}
    all_channels = set(channels.keys())

    if HF_settings.get("mode") == "debug":
        print(f"polygons_per_channel.keys() = {polygons_per_channel.keys()}, \
                list(annotated_channels) = {list(annotated_channels)}")

    # Iterate over valid channel combinations (per Z-frame if n>1)
    for combo in HF_utils.power_set(annotated_channels, n, c):
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

        if HF_settings.get("mode") == "debug":
            print(f"Image {base}, combo = {combo}, \
                    noise_channels = {noise_channels}, \
                    noise_candidates = {noise_candidates}")

        # Compose hue-fused RGB and save
        palette = HF_palette.get_random_palette(hash_data=filename)
        img_rgb = HF_ImageOps.compose_hue_fusion(
            channels, combo, palette, noise_channels=noise_channels
        )
        img_path = os.path.join(img_dir, filename)
        Image.fromarray(img_rgb).save(img_path, "JPEG")

        # Flatten annotations for the selected channels and write YOLO labels.
        annotations = list(chain.from_iterable(polygons_per_channel.get(ch, []) for ch in combo))
        if annotations:
            label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
            HF_utils.save_yolo_segmentation_label(label_path, annotations, class_ids)



def split_train_val():
    """
    Split the dataset into training and validation subsets with two guarantees:

    1) Group integrity: all JPEGs derived from the same TIFF (including any
       "_MIP" variants and channel combinations) are assigned together to
       either train or validation.

    2) Class representation: choose validation groups to approximate the
       global class distribution (greedy selection toward per-class targets),
       instead of taking a naive random sample of individual images.

    Heuristic
    ---------
    - Build groups of images by recovering the TIFF base name from each JPEG:
      strip the trailing "_<num>[_<num>]..." channel suffix, and any optional
      "_MIP" immediately before that suffix.
    - For each group, aggregate a vector of class counts from its label files.
    - Let `validation_frac` be p. Set per-class targets to p × (global class counts).
    - Greedily pick the next group that most reduces the L2 distance between
      current validation class counts and the per-class targets, until the
      validation image budget (≈ p × total images) is reached.

    Paths
    -----
    Reads from:
      - dataset/images/train/*.jpg
      - dataset/labels/train/*.txt
    Moves selected groups to:
      - dataset/images/val/
      - dataset/labels/val/

    Notes
    -----
    - If some classes are extremely rare, perfect stratification may be
      impossible with whole-group selection; this routine still tries to
      preserve them in both splits when feasible.
    - Missing label files are skipped with a warning.
    """
    import os, re, random, shutil
    from glob import glob
    from collections import defaultdict, Counter

    img_dir = HF_folders.get_image_train_dir()
    lbl_dir = HF_folders.get_label_train_dir()
    img_val_dir = HF_folders.get_image_val_dir()
    lbl_val_dir = HF_folders.get_label_val_dir()

    # --- 1) Discover images and group them by their originating TIFF ----------
    image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if not image_paths:
        HF_log.warn("No training images found to split")
        return

    # Regex: remove the trailing "_<num>[_<num>]*" (channel combo),
    # with an optional "_MIP" right before it → yields the TIFF base.
    # Examples:
    #   my_sample_A_3_4_5.jpg       → group "my_sample_A"
    #   my_sample_A_MIP_1_2.jpg     → group "my_sample_A"
    stem_re = re.compile(r'^(?P<stem>.*?)(?:_MIP)?(?:_\d+(?:_\d+)*)$')

    groups = defaultdict(list)  # base → [image_path, ...]
    for ip in image_paths:
        name = os.path.splitext(os.path.basename(ip))[0]
        m = stem_re.match(name)
        base = m.group("stem") if m else name
        groups[base].append(ip)

    # --- 2) Build per-group class counts from YOLO label files -----------------
    # We only need class IDs; we don't care about polygon geometry here.
    group_class_counts = {}
    total_class_counts = Counter()
    total_images = 0

    for base, imgs in groups.items():
        cls_counts = Counter()
        for ip in imgs:
            img_name = os.path.basename(ip)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            lp = os.path.join(lbl_dir, label_name)
            if not os.path.isfile(lp):
                HF_log.warn(f"Missing label for {img_name}; skipping in stats")
                continue
            with open(lp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # First token is the class id
                    try:
                        cls_id = int(line.split()[0])
                        cls_counts[cls_id] += 1
                        total_class_counts[cls_id] += 1
                    except Exception:
                        # Ignore malformed label lines
                        continue
        group_class_counts[base] = cls_counts
        total_images += len(imgs)

    # --- 3) Compute targets for validation ------------------------------------
    frac = float(HF_settings.get("validation_frac") or 0.2)
    target_images = max(1, int(round(total_images * frac)))  # approximate budget
    target_per_class = {k: v * frac for (k, v) in total_class_counts.items()}

    # --- 4) Greedy selection of groups to approach class targets --------------
    remaining = set(groups.keys())
    val_groups = set()
    cur_class = Counter()
    cur_images = 0

    # Randomize processing order to avoid pathological choices on ties.
    pool = list(remaining)
    random.shuffle(pool)

    def score_if_add(base):
        """L2 distance to targets if we add this base."""
        tmp = cur_class.copy()
        tmp.update(group_class_counts.get(base, {}))
        # Sum of squared residuals across known classes
        sse = 0.0
        for k, tgt in target_per_class.items():
            diff = tmp.get(k, 0) - tgt
            sse += diff * diff
        # Lightly penalize overshooting the image budget
        img_over = max(0, (cur_images + len(groups[base])) - target_images)
        return sse + 0.01 * (img_over ** 2)

    # Pick groups until we reach the (approximate) image budget.
    while pool and cur_images < target_images:
        # Choose the group that yields the best score improvement
        best_base = min(pool, key=score_if_add)
        val_groups.add(best_base)
        cur_class.update(group_class_counts.get(best_base, {}))
        cur_images += len(groups[best_base])
        pool.remove(best_base)

    # Remaining groups go to training.
    train_groups = remaining - val_groups

    # --- 5) Move files on disk -------------------------------------------------
    def move_group(baseset, dest_img_dir, dest_lbl_dir):
        for base in baseset:
            for ip in groups[base]:
                img_name = os.path.basename(ip)
                label_name = os.path.splitext(img_name)[0] + ".txt"
                lp = os.path.join(lbl_dir, label_name)

                dst_img = os.path.join(dest_img_dir, img_name)
                dst_lbl = os.path.join(dest_lbl_dir, label_name)

                # Move image
                shutil.move(ip, dst_img)

                # Move label if present (may be missing for some images)
                if os.path.isfile(lp):
                    shutil.move(lp, dst_lbl)
                else:
                    HF_log.warn(f"Label missing for {img_name}; moved image only")

    # Ensure destinations exist (they should, but be defensive)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    move_group(val_groups, img_val_dir, lbl_val_dir)
    # train_groups remain in-place under images/labels/train

    HF_log.info(
        f"Split complete: {len(val_groups)} TIFF groups → val "
        f"({cur_images} images, target ≈ {target_images}); "
        f"{len(train_groups)} groups remain in train."
    )




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
    size = HF_settings.get("size")
    assert (stacked_channels[0].shape == (size, size))

    masks_dir = HF_folders.get_masks_dir()
    contours_dir = HF_folders.get_contours_dir()

    # For each channel of the MIP, merge polygons from all Z-slices
    polygons_per_stacked_channel = defaultdict(list)
    for ch in range(c):
        # 1-based indices of frames for this channel across Z: 1, 1+c, 1+2c, ...
        indices = [j + 1 for j in range(ch, n * c, c)]  # 1, 1+c, 1+2c, ...
        
        # Collect per-slice polygons for the channel: [(label, [polys...]), ...]
        polygons_subset = [polygons_per_channel.get(idx, []) for idx in indices]

        # Build a fused mask per class from all Z-slice polygons
        allowed_items = [(x, y) for x, y in class_ids.items() if HF_ImageInfo.allows_MIP_generation(x)]
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
                        pts = HF_geometry.flat_to_pts_xy(flat)   # (N, 2)
                        if pts.shape[0] >= 3:
                            all_polys_px.append(pts)

            # If nothing accumulated, skip this class/channel
            if not all_polys_px:
                continue

            # Fused binary mask for the class on the MIP channel
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(mask, all_polys_px, 255)

            clean_mask = HF_segmentation.noise_and_gaps(mask)
            # Persist the fused class mask (for QA or reuse)
            mask_path = os.path.join(masks_dir, f"{base}_MIP_{class_name}_mask.png")
            cv2.imwrite(mask_path, clean_mask)
            # Extract refined contours and convert to YOLO polygons
            final_contours = HF_segmentation.mask_to_polygons(clean_mask)
            yolo_polygons = HF_geometry.contours_to_yolo_polygons(final_contours)
 
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
    data_dir = HF_settings.get("tiff_dir")
    image_paths = sorted(glob(os.path.join(data_dir, "*.tif")))
    
    class_ids = HF_settings.load_class_definitions()
    HF_folders.write_yolo_yaml(class_ids)
    HF_ImageInfo.initialize()

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        HF_ImageInfo.set_current_image(img_name)
        img_base = HF_ImageInfo.get_current_base()

        if not HF_ImageInfo.image_has_instructions():
            HF_log.warn(f"Skipping file {img_name} - no annotations")
            continue

        img = tifffile.imread(img_path)
        if not HF_geometry.is_valid_image_format(img):
            HF_log.warn(f"Skipping file {img_name}, wrong shape {img.shape}")
            continue

        # Resize channels to the configured size; get ratio and (n, c)
        channels, ratio, (n, c) = HF_ImageOps.resize_multichannel_image(img)   
        polygons_per_channel = prepare_class_inputs(channels, n, c, ratio)
        
        # QA overlays then dataset generation
        generate_contours(img_base, polygons_per_channel, channels, class_ids)     
        generate_dataset(img_base, n, c, channels, polygons_per_channel)

        # Optional MIP export when multiple Z-slices exist
        # FIXME: currently deactivated because it is not satisfactory.
        if n > 1 and False:
            max_intensity_projection_multichannel(
                img_name, img_base, img, polygons_per_channel,
                class_ids, n, c, ratio
            )

    # Split train/val at the end so both single-plane and MIP outputs are included
    split_train_val()
