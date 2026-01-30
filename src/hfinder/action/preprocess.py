"""
High-level preprocessing pipeline for converting multichannel TIFF images
into a YOLOv8-compatible segmentation dataset.

This module implements the full annotation workflow: mask generation,
polygon extraction, geometric refinement, visual QA rendering, and final
export of RGB training images together with YOLO segmentation labels.
Only well-defined, channel-wise training samples are produced; the former
hue-fusion and noise-channel strategies have been removed for clarity and
predictability.

Overview
--------
1.  Load class-specific directives for the current image
    (channel assignments, thresholds, optional manual JSON segmentations).

2.  Generate binary masks for each annotated class and channel, using
        • fixed thresholds,
        • manual polygon files,
        • or automatic thresholding.
    Masks are saved for inspection.

3.  Convert masks or manual annotations into YOLO-style normalized
    polygons. Coordinates are rescaled using the resize ratio provided by
    :mod:`HF_ImageOps`.

4.  Apply geometric post-processing:
        • contour simplification via Douglas–Peucker,
        • uniform arc-length resampling of polygon vertices.
    The parameters (`epsilon_rel`, `min_points`, `target_points`)
    are retrieved from :data:`HF_settings` to ensure global consistency.

5.  Render visual QA overlays (filled polygons + class labels) for each
    channel, enabling rapid inspection of annotation quality.

6.  Export one RGB training image per channel (grayscale is replicated
    to 3 channels when necessary), together with the corresponding YOLO
    `.txt` labels. Each channel is treated as an independent sample.

7.  Optionally construct a Max-Intensity-Projection (MIP) mini-dataset
    for Z-stacks. This pathway remains experimental and is currently not
    enabled by default.

8.  After all images have been processed, perform a stratified
    train/validation split via :mod:`dataset_split`, preserving class
    diversity across subsets.

Public API
----------
- generate_masks_and_polygons(channels, n, c, ratio, allowed_category_names)
      Produce binary masks and initial polygons for each class/channel.

- simplify_and_resample_polygons(polygons_per_channel)
      Apply geometric simplification and resampling rules.

- generate_contours(base, polygons_per_channel, channels, classes)
      Render visual QA overlays.

- export_yolo_images_and_labels(base, channels, polygons_per_channel)
      Export per-channel RGB samples and their YOLO segmentation labels.

- build_mip_dataset_from_stack(...)
      Experimental MIP-based dataset construction.

- build_full_training_dataset()
      Main high-level routine converting all TIFFs in ``tiff_dir`` into a
      structured YOLO dataset.

Notes
-----
- Channels remain 1-based throughout, matching the UI conventions.
- Polygon coordinates are normalized to the resized image dimensions.
- JSON segmentations are currently supported only for single-plane images.
- Degenerate or very small polygons may be removed during post-processing.
"""



import os
import cv2
import json
import random
import tifffile
import colorsys
import numpy as np
from PIL import Image
from glob import glob
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
from hfinder.action import dataset_split as HF_DatasetSplit


def simplify_flat_polygon(flat, epsilon_rel, min_points):
    """
    Simplify a normalized YOLO-style polygon using the Douglas–Peucker algorithm.

    The polygon is given as a flat list of normalized coordinates
    ``[x1, y1, ..., xn, yn]`` in the range ``[0.0, 1.0]``. The relative
    tolerance ``epsilon_rel`` is applied to the normalized perimeter of the
    polygon; larger values produce coarser, less detailed contours. If the
    simplified polygon would contain fewer than ``min_points`` vertices, the
    original polygon is returned unchanged.

    :param flat: Flattened list of polygon vertices as
                 ``[x1, y1, ..., xn, yn]`` in normalized coordinates.
    :type flat: list[float]
    :param epsilon_rel: Relative tolerance used by the Douglas–Peucker
                        simplification, expressed as a fraction of the
                        polygon perimeter.
    :type epsilon_rel: float
    :param min_points: Minimum number of vertices required for the simplified
                       polygon; polygons falling below this threshold are
                       returned unchanged.
    :type min_points: int
    :return: Simplified polygon as a flattened list of vertices, or the
             original polygon if simplification is not applicable.
    :rtype: list[float]
    """
    if not flat or len(flat) < 2 * min_points:
        return flat

    pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    cnt = pts.reshape(-1, 1, 2)

    # longueur en coordonnées normalisées (indépendante de la taille en pixels)
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return flat

    eps = epsilon_rel * peri
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

    if approx.shape[0] < min_points:
        # trop simplifié -> on garde la version d'origine
        return flat

    return approx.flatten().tolist()



def resample_flat_polygon(flat, target_points):
    """
    Resample a normalized polygon so that vertices are evenly spaced
    along its perimeter.

    The polygon is given as a flat list of normalized coordinates
    ``[x1, y1, ..., xn, yn]`` in the range ``[0.0, 1.0]``. The function
    constructs a closed contour and interpolates new vertices at
    (approximately) regular arc-length intervals, returning a polygon
    with ``target_points`` vertices.

    :param flat: Flattened list of polygon vertices as
                 ``[x1, y1, ..., xn, yn]`` in normalized coordinates.
    :type flat: list[float]
    :param target_points: Desired number of vertices in the resampled
                          polygon. Very small polygons may effectively
                          limit the useful number of points.
    :type target_points: int
    :return: Resampled polygon as a flattened list of vertices with
             (approximately) ``target_points`` points.
    :rtype: list[float]
    """
    if not flat or len(flat) < 6:
        return flat

    pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    n = pts.shape[0]
    if n < 2:
        return flat

    # on ferme explicitement le polygone
    pts_closed = np.vstack([pts, pts[0]])

    # distances cumulées le long du contour
    seg = np.sqrt(((np.diff(pts_closed, axis=0)) ** 2).sum(axis=1))
    d = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = d[-1]
    if perimeter <= 0:
        return flat

    # nb de points utile (évite de gonfler un petit contour)
    K = min(target_points, max(n, 4))
    step = perimeter / K

    new_pts = []
    for k in range(K):
        pos = k * step
        # position sur la chaîne de segments
        idx = np.searchsorted(d, pos) - 1
        if idx < 0:
            idx = 0
        if idx >= len(pts):
            idx = len(pts) - 1

        t_den = d[idx + 1] - d[idx] if (idx + 1 < len(d)) else 1.0
        if t_den <= 1e-8:
            t = 0.0
        else:
            t = (pos - d[idx]) / t_den

        p = (1.0 - t) * pts_closed[idx] + t * pts_closed[idx + 1]
        new_pts.append(p)

    new_pts = np.asarray(new_pts, dtype=np.float32)
    return new_pts.flatten().tolist()



def simplify_and_resample_polygons(polygons_per_channel):
    """
    Apply geometric post-processing (simplification and resampling)
    to all polygons in all channels.

    This function iterates over the mapping of channels to class-labeled
    polygons, applies :func:`simplify_flat_polygon` followed by
    :func:`resample_flat_polygon` to each polygon, and returns a new
    mapping with optimized polygons. Very small or degenerate polygons
    may be discarded, and channels for which no valid polygons remain
    are omitted from the result.

    The numerical parameters controlling simplification and resampling
    (e.g. ``"epsilon_rel"``, ``min_points"``,
    ``"target_points"``) are read from :data:`HF_settings` inside
    this function, so they do not appear in its public signature and
    their default values are defined in a single place.

    The expected input structure is::

        {
            channel_id: [
                (class_name, [flat_polygon_1, flat_polygon_2, ...]),
                ...
            ],
            ...
        }

    where each ``flat_polygon_i`` is a flattened list of normalized
    coordinates ``[x1, y1, ..., xn, yn]``.

    :param polygons_per_channel: Mapping from channel identifiers to
                                 lists of ``(class_name, [polygons])``,
                                 where each polygon is a flattened list
                                 of normalized vertices.
    :type polygons_per_channel: dict[Any, list[tuple[str, list[list[float]]]]]
    :return: New mapping with simplified and resampled polygons, using
             the same structure as the input.
    :rtype: dict[Any, list[tuple[str, list[list[float]]]]]
    """
    epsilon_rel = HF_settings.get("epsilon_rel")
    target_points = HF_settings.get("target_points")
    min_points = HF_settings.get("min_points")
    
    new_map = defaultdict(list)

    for ch, items in polygons_per_channel.items():
        for class_name, polys in items:
            new_polys = []
            for flat in polys:
                if not flat or len(flat) < 2 * min_points:
                    continue  # ignore small artefacts

                flat_simpl = simplify_flat_polygon(flat, epsilon_rel, min_points)
                flat_resampled = resample_flat_polygon(flat_simpl, target_points)
                new_polys.append(flat_resampled)

            if new_polys:
                new_map[ch].append((class_name, new_polys))

    return new_map



def generate_masks_and_polygons(channels, n, c, ratio, allowed_category_names):
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
    allowed = set(allowed_category_names) if allowed_category_names else None

    for cls in HF_ImageInfo.get_classes():
        if allowed is not None and cls not in allowed:
            continue
 
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
                cv2.imwrite(binary_output, binary)

        elif poly_json is not None:
            # Load user-provided segmentation polygons (single-plane only)
            if n > 1:
                HF_log.fail(f"File '{base}.tif' - applying user segmentation to Z-stacks has not been implemented yet")
            json_path = os.path.join(HF_settings.get("tiff_dir"), poly_json)
            polygons = HF_segmentation.channel_custom_segment(json_path, ratio, {cls})
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
                cv2.imwrite(binary_output, binary)

    return results



def generate_contours(base, polygons_per_channel, channels, classes):
    """
    Draw filled/outlined polygons over grayscale channels and save overlays.

    :param base: Base name used for output files.
    :type base: str
    :param polygons_per_channel: Map channel → [(class_name, [polys...]), ...].
    :type polygons_per_channel: dict[int, list[tuple[str, list[list[float]]]]]
    :param channels: Dict channel → 2D grayscale image.
    :type channels: dict[int, np.ndarray]
    :param classes: Class names.
    :type classes: list[str]
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
            if class_name not in classes:
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



def export_yolo_images_and_labels(base, channels, polygons_per_channel):
    """
    Export per-channel RGB images and their corresponding YOLO segmentation labels.

    Each entry in `channels` is treated as an independent training sample.
    Grayscale images are converted to RGB by channel replication, while
    already-RGB arrays are preserved as is. For every channel, a JPEG image
    and a YOLO-formatted `.txt` annotation file are created. If no polygons
    are associated with a given channel, an empty annotation file is written
    to maintain dataset integrity.

    Parameters
    ----------
    base : str
        Basename (without extension) used to construct output filenames.
    channels : dict[int, np.ndarray]
        Mapping from channel index to a 2D or 3D NumPy array representing
        the corresponding image slice.
    polygons_per_channel : dict[int, list[tuple[str, list[list[float]]]]]
        Mapping from channel index to the list of annotated objects.
        Each object is represented as `(class_name, polygon_list)`, where
        `polygon_list` contains one or more polygons expressed as sequences
        of `(x, y)` coordinates.

    Output
    ------
    Writes one JPEG image and one YOLO `.txt` file per channel into the
    directory specified by HF_Settings. Filenames follow the pattern:
        <base>_cXX.jpg
        <base>_cXX.txt
    """
    img_dir = HF_folders.get_image_train_dir()
    lbl_dir = HF_folders.get_label_train_dir()
    classes = HF_settings.load_class_list()

    for ch, img2d in channels.items():

        filename = f"{os.path.splitext(base)[0]}_c{ch:02d}.jpg"
        img_path = os.path.join(img_dir, filename)

        if img2d.ndim == 2: # grayscale image.
            img_rgb = np.stack([img2d]*3, axis=-1).astype(np.uint8)
        else: # let's assume it is RGB.
            img_rgb = img2d
        Image.fromarray(img_rgb).save(img_path, "JPEG")

        # Write labels (can be empty files if there are no annotations).
        annotations = polygons_per_channel.get(ch, [])
        label_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
        HF_utils.save_yolo_segmentation_label(label_path, annotations, classes)



def build_mip_dataset_from_stack(img_name, base, stack, polygons_per_channel, class_ids, n, c, ratio):
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

    export_yolo_images_and_labels(
        base + "_MIP",
        channels=stacked_channels_dict,
        polygons_per_channel=polygons_per_stacked_channel
    )



def build_full_training_dataset():
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

    # Retrieve all TIFFS, but may filter out some if a manifest file is present.
    image_paths = sorted(glob(os.path.join(data_dir, "*.tif")) + glob(os.path.join(data_dir, "*.tiff")))
    manifest_path = HF_settings.get("manifest")
    manifest = None

    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = set(json.load(f))
        image_paths = [p for p in image_paths if os.path.basename(p) in manifest]
        HF_log.info(f"Manifest enabled: {len(image_paths)} images kept from {manifest_path}")

        # Report any missing TIFF file compared to the manifest.
        missing = sorted(manifest - {os.path.basename(p) for p in image_paths})
        if missing:
            HF_log.warn(f"{len(missing)} manifest images not found in tiff_dir (showing up to 10): {missing[:10]}")


    classes = HF_settings.load_class_list()
    allowed_category_names = set(classes)
    #HF_log.info(f"Active classes: {classes}")
    HF_folders.write_yolo_yaml(classes)
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
        polygons_per_channel = generate_masks_and_polygons(channels, n, c, ratio, allowed_category_names)
        
        # Post-traitement des polygones pour simplifier et homogénéiser les contours
        polygons_per_channel = simplify_and_resample_polygons(polygons_per_channel)

        # QA overlays then dataset generation
        generate_contours(img_base, polygons_per_channel, channels, classes)     
        export_yolo_images_and_labels(img_base, channels, polygons_per_channel)

        # Optional MIP export when multiple Z-slices exist
        # FIXME: currently deactivated because it is not satisfactory.
        if n > 1 and False:
            build_mip_dataset_from_stack(
                img_name, img_base, img, polygons_per_channel,
                classes, n, c, ratio
            )

    # Split train/val at the end so both single-plane and MIP outputs are included
    HF_DatasetSplit.split_train_val()
