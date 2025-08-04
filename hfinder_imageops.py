import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
import hfinder_palette as HFinder_palette
import hfinder_settings as HFinder_settings



def is_valid_image_format(img):
    """
    Check whether a NumPy array corresponds to a valid multichannel or z-stack 
    image. Supported image formats are:
        - 3D multichannel images with shape (C, H, W)
        - 4D z-stacks with shape (Z, C, H, W)
    Images must have fewer than 10 channels and spatial dimensions greater than 
    64×64.

    :param img: Input image array.
    :type img: np.ndarray
    :returns: True if the image matches a supported format, False otherwise.
    :rtype: bool
    """
    valid_ndim = False
    if img.ndim == 4:
        valid_ndim = True
        _, c, h, w = img.shape
    elif img.ndim == 3:
        valid_ndim = True
        c, h, w = img.shape
    return valid_ndim and c < 10 and h > 64 and w > 64



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
 
    if img.ndim == 4:
        n, c, h, w = img.shape
    else:
        n = 1
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



def colorize_with_hue(frame, hue):
    norm = frame.astype(np.float32)
    norm /= norm.max() if norm.max() > 0 else 1

    h = np.full_like(norm, hue)
    s = np.ones_like(norm)
    v = norm

    hsv = np.stack([h, s, v], axis=-1)  # shape (H, W, 3)
    rgb = hsv_to_rgb(hsv)
    return rgb



def compose_hue_fusion(channels, selected_channels, palette, noise_channels=None):
    """
    Compose an RGB image by blending each selected channel with a random hue.
    """
    h, w = next(iter(channels.values())).shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    for ch in selected_channels:
        frame = channels[ch]
        hue = HFinder_palette.get_color(ch, palette=palette)[0]
        colored = colorize_with_hue(frame, hue) 
        rgb += colored

    # Ajout de bruit visuel contrôlé
    if noise_channels:
        for ch in noise_channels:
            frame = channels[ch]
            hue = HFinder_palette.get_color(ch, palette=palette)[0]
            noise = colorize_with_hue(frame, hue)
            rgb += 0.3 * noise 

    # Clip to [0, 1] in case of saturation, then scale to [0, 255]
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)
