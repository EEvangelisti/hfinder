import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tifffile import imread
from random import shuffle, sample
from matplotlib.colors import hsv_to_rgb
import hfinder_log as HFinder_log
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings


# TODO: Handle regular TIFF, z-stacks and time stacks properly.
def is_channel_first(img, hyphae_channel):
    if img.ndim != 3:
        return False
    c, h, w = img.shape
    # check if hyphae_channel fits and the other dims are likely spatial
    return hyphae_channel < c and h > 64 and w > 64



def load_json(json_path):
    """
    Loads a JSON file containing hyphae channel mappings.

    Parameters
    ----------
    json_path : str
        Path to the JSON file that maps image filenames to their hyphae channel indices.

    Returns
    -------
    dict
        A dictionary mapping image filenames (str) to channel indices (int),
        typically 1-based (e.g., {"img1.tif": 2, "img2.tif": 1}).

    Notes
    -----
    - The JSON file must be a valid mapping from strings to integers.
    - Channel indices are expected to be 1-based for compatibility with downstream code.
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

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
    import cv2
    from numpy import empty

    target_size = HFinder_settings.get("target_size")
    c, h, w = img.shape
    resized = empty((c, *target_size), dtype=img.dtype)
    for i in range(c):
        resized[i] = cv2.resize(img[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    ratios = tuple(x / w for x in target_size)
    assert(ratios[0] == ratios[1])
    return resized, ratios[0]

def make_binary_mask(img, hyphae_channel):
    """
    Generates a binary mask from the specified hyphae channel of a multichannel image
    and extracts the corresponding external contours as YOLO-style normalized polygons.

    Parameters
    ----------
    img : np.ndarray
        A 3D NumPy array of shape (C, H, W), where C is the number of channels.
    hyphae_channel : int
        Index (1-based) of the channel corresponding to hyphae. For example, 
        `1` targets the first channel.

    Returns
    -------
    binary : np.ndarray
        A 2D binary mask (dtype uint8) where pixels above the 90th percentile are set to 255.
    yolo_polygons : list of list of float
        A list of polygons, each represented as a flat list of normalized coordinates:
        [x1, y1, x2, y2, ..., xn, yn], with all values ∈ [0, 1].

    Notes
    -----
    - Only external contours are extracted (using `cv2.RETR_EXTERNAL`).
    - Contours with fewer than 3 points are ignored.
    - Coordinates are normalized by dividing by the image width and height respectively.
    """
    hyphae = img[hyphae_channel - 1, :, :]
    _, h, w = img.shape
    thresh_val = np.percentile(hyphae, 90)
    _, binary = cv2.threshold(hyphae, thresh_val, 255, cv2.THRESH_BINARY)
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



def colorize_with_hue(frame, hue):
    """
    Applies a constant hue to a grayscale frame to produce a pseudo-colored RGB image.

    The input frame is normalized and combined with fixed saturation and the provided hue
    to form an HSV image, which is then converted to RGB.

    Args:
        frame (np.ndarray): 2D grayscale image (H, W).
        hue (float): Hue value in [0, 1) to apply (0 = red, 1/3 = green, 2/3 = blue, etc.).

    Returns:
        np.ndarray: RGB image (H, W, 3) with the input intensity mapped to the given hue.
    """
    norm = frame.astype(np.float32)
    norm /= norm.max() if norm.max() > 0 else 1

    h = np.full_like(norm, hue)
    s = np.ones_like(norm)
    v = norm

    hsv = np.stack([h, s, v], axis=-1)  # shape (H, W, 3)
    rgb = hsv_to_rgb(hsv)
    return rgb



def make_random_rgb_composites(img, hyphae_channel, n=10):
    """
    Creates N randomized RGB composites from a multichannel image, emphasizing 
    the hyphae channel.

    For each composite:
    - The hyphae channel is combined with 1 or 2 randomly selected other channels.
    - Each channel is colorized using a random hue and added to form an RGB image.
    - The RGB order is randomly shuffled.

    Args:
        img (np.ndarray): Multichannel image with shape (C, H, W).
        hyphae_channel (int): 1-based index of the channel containing hyphae.
        n (int): Number of composites to generate.

    Returns:
        List[np.ndarray]: List of RGB images (H, W, 3) with float32 values ∈ [0,1].
    """
    hyphae_channel -= 1 # we assume the user counts from 1.
    channels = list(range(img.shape[0]))
    channels.remove(hyphae_channel)

    composites = []
    for _ in range(n):
        num_extra = np.random.choice(len(channels))
        others = sample(channels, num_extra)
        combo = [hyphae_channel] + others
        shuffle(combo)  # ordre aléatoire dans RGB

        rgb = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
        for ch in combo:
            hue = np.random.rand()  # random hue in [0, 1)
            frame = img[ch, :, :]
            colored = colorize_with_hue(frame, hue)
            rgb += colored  # additive mixing

        composites.append(rgb)

    shuffle(composites)
    return composites



def write_yolo_labels(label_output, yolo_polygons, class_id=0):
    """
    Writes YOLO-style segmentation labels for a given image.

    Parameters:
        image_path (str): Full path to the image file (used to derive label filename).
        yolo_polygons (list of list of floats): List of polygons, each represented 
        by a list of normalized coordinates [x1, y1, x2, y2, ..., xn, yn].
        label_dir (str): Path to the directory where the label file should be saved.
        class_id (int): Integer representing the object class (default: 0).
    """
    with open(label_output, "w") as f:
        for polygon in yolo_polygons:
            if len(polygon) < 6:
                continue  # Skip invalid (less than 3 points)
            coords = " ".join(f"{x:.6f}" for x in polygon)
            f.write(f"{class_id} {coords}\n")



def generate_dataset(folder_tree):
    """
    Generates a training dataset for YOLO-style segmentation from multichannel TIFF images.

    This function processes all `.tif` files in the specified directory. For each image, it:
    - Retrieves the hyphae channel from a JSON mapping.
    - Computes a binary mask of hyphae and their contours.
    - Saves the mask as a PNG image.
    - Generates multiple random RGB composites from the TIFF.
    - Splits the composites into training and validation sets.
    - Saves image paths and mask paths into the `folder_tree` structure.

    Args:
        folder_tree (dict): The folder and file registry used to track dataset structure.

    Notes:
        - TIFF files not listed in the JSON mapping are skipped.
        - If a TIFF is not multichannel or unreadable, it is skipped with a warning.
        - The function uses `image_id` and composite index `i` to name output files uniquely.
    """
    image_id = 0

    tiff_dir = HFinder_settings.get("tiff_dir")
    channels = HFinder_settings.get("channels")
    json_path = os.path.join(tiff_dir, channels)
    hyphae_map = load_json(json_path)
    if hyphae_map is None:
        HFinder_log.fail(f"Missing file {channels}")

    thresholds = HFinder_settings.get("thresholds")
    json_path = os.path.join(tiff_dir, thresholds)
    thresold_map = load_json(json_path)

    tiff_files = glob(os.path.join(tiff_dir, '*.tif'))
    shuffle(tiff_files)

    for tiff_path in tiff_files:

        tiff_path = os.path.abspath(tiff_path)
        filename = os.path.basename(tiff_path)
        base = os.path.splitext(filename)[0]

        if filename not in hyphae_map:
            HFinder_log.warn(f"Skipping file {filename} - no entry in {channels}")
            continue

        hyphae_channel = hyphae_map[filename]
        img = None
        
        try:
        
            img = imread(tiff_path)
            if not is_channel_first(img, hyphae_channel):
                HFinder_log.warn(f"Skipping file {filename} - not a multichannel image")
                continue

        except Exception as e:
            HFinder_log.warn(f"Skipping file {filename} - {e}")
            continue

        if img is not None:

            image_id += 1

            img, ratio = resize_multichannel_image(img)
            HFinder_log.info(f"Image '{filename}' resized by factor {round(ratio, 2)}")

            # (1) check whether annotations exist already!
            # (2) check whether there is a custom threshold.
            # (3) apply the standard method below.
            # First, get binary mask and YOLO polygons.
            binary, yolo_polygons = make_binary_mask(img, hyphae_channel)
            binary_output = os.path.join(folder_tree["root"], f"dataset/masks/{base}.mask.png")
            plt.imsave(binary_output, binary, cmap='gray')
            HFinder_folders.append_subtree(folder_tree, "dataset/masks", binary_output)

            # Then, generate multiple composite images.
            composites = make_random_rgb_composites(img, hyphae_channel)

            for i, rgb in enumerate(composites):

                # Composite images
                image_key = "dataset/images/train" if i <= 7 else "dataset/images/val"               
                composite_output = os.path.join(folder_tree["root"], f"{image_key}/image-{image_id}-{i}.png")
                plt.imsave(composite_output, np.clip(rgb, 0, 1))
                HFinder_folders.append_subtree(folder_tree, image_key, composite_output)

                # Corresponding labels
                label_key = "dataset/labels/train" if i <= 7 else "dataset/labels/val"
                label_output = os.path.join(folder_tree["root"], f"{label_key}/image-{image_id}-{i}.txt")
                write_yolo_labels(label_output, yolo_polygons)
                HFinder_folders.append_subtree(folder_tree, label_key, label_output)


