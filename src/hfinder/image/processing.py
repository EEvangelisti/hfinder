"""
Image processing utilities for HFinder.

This module provides helper functions to:
- Validate image array formats before processing.
- Resize multi-channel or z-stack images to a target size, preserving channels.
- Apply hue-based false coloring to individual channels.
- Compose RGB fusion images from selected channels with optional noise channels.

Public API
----------
- is_valid_image_format(img): Check if an array is a supported image format.
- resize_multichannel_image(img): Resize a multi-channel/z-stack image.
- colorize_with_hue(frame, hue): Apply hue-based coloring to a single channel.
- compose_hue_fusion(channels, selected_channels, palette, noise_channels=None):
    Compose an RGB fusion from multiple channels using a color palette.

Notes
-----
- Images are assumed to be NumPy arrays.
- Channel indexing in returned dicts is 1-based for consistency with
  image annotation JSON files.
- Colorization uses HSV → RGB conversion with full saturation and per-pixel value.
"""

import cv2
import matplotlib
import numpy as np
from PIL import Image
from hfinder.core import log as HF_log
from hfinder.core import palette as HF_palette
from hfinder.session import settings as HF_settings
from hfinder.session import current as HF_ImageInfo



def normalize_channel(arr: np.ndarray, mode: str = "uint8") -> np.ndarray:
    """
    Normalize a single-channel image array.

    :param arr: Input 2D image.
    :type arr: np.ndarray
    :param mode: Normalization mode:
        - "float": scale to [0.0, 1.0] as float32
        - "uint8": scale to [0, 255] as uint8 (default mode)
        - "none": return unchanged if already compatible
    :type mode: str
    :return: Normalized array according to mode.
    :rtype: np.ndarray
    :raises ValueError: If mode is unsupported.
    """
    if mode == "uint8" and arr.dtype == np.uint8:
        return arr
    if mode == "float" and arr.dtype == np.float32 and arr.min() >= 0.0 and arr.max() <= 1.0:
        return arr

    a = arr.astype(np.float32, copy=False)
    vmin, vmax = float(a.min()), float(a.max())

    if vmax <= vmin:
        a = np.zeros_like(a, dtype=np.float32)
    else:
        a = (a - vmin) / (vmax - vmin)

    if mode == "float":
        return a
    elif mode == "uint8":
        return (a * 255.0).astype(np.uint8)
    elif mode == "none":
        return arr
    else:
        raise ValueError(f"Unsupported mode: {mode}")




def extract_frame(img, ch=0, z=0, norm=normalize_channel):
    """
    Return a 2D plane (H, W) from an array shaped (C, H, W) or (Z, C, H, W).

    Selection rules:
      - (C, H, W): select channel ``ch`` (default 0).
      - (Z, C, H, W): select slice ``z`` (default 0) and channel ``ch`` (default 0).

    :param img: Image array.
    :type img: numpy.ndarray
    :param ch: Channel index to select (defaults to 0 if None).
    :type ch: int | None
    :param z: Z index to select (only used when Z is present; defaults to 0 if None).
    :type z: int | None
    :returns: 2D image plane as (H, W).
    :rtype: numpy.ndarray
    :raises ValueError: If the array does not match (C,H,W) or (Z,C,H,W).
    :raises AssertionError: If indices are out of bounds.
    """
    norm = (lambda x: x) if norm is None else norm
    if img.ndim == 2:
        return norm(img)
    elif img.ndim == 3:
        C, H, W = img.shape
        assert (0 <= ch < C), f"Channel index {ch} out of range [0, {C-1}] for shape {img.shape}."
        return norm(img[ch, :, :])
    elif img.ndim == 4:
        Z, C, H, W = img.shape
        assert (0 <= z < Z), f"Z index {z} out of range [0, {Z-1}] for shape {img.shape}."
        assert (0 <= ch // Z < C), f"Channel index {ch} out of range [0, {C-1}] for shape {img.shape}."
        return norm(img[z, ch // Z, :, :])
    else:
        raise ValueError(f"Unsupported shape {img.shape}; expected (C, H, W) or (Z, C, H, W).")



def resize_multichannel_image(img):
    """
    Resize a multi-channel or z/t-stack image to the configured square size.

    Uses bilinear interpolation (cv2.INTER_LINEAR) per channel.

    :param img: Input image.
        Shape (C, H, W) for multi-channel or (Z, C, H, W) for z/t-stack.
    :type img: np.ndarray
    :return: Tuple (channels_dict, ratio, dims):
        - channels_dict: dict[int, np.ndarray] mapping 1-based channel indices
          to resized 2D arrays of shape (size, size).
        - ratio: Width scaling factor (size / original_width).
        - dims: Tuple (n, c) where:
            n = number of z/t-slices (1 for single plane),
            c = number of channels per slice.
    :rtype: tuple
    """

    size = HF_settings.get("size")
 
    # For Z-stacks, (Z, C, H, W) → max intensity projection over Z → (C, H, W)
    if img.ndim == 4:
        HF_log.info(f"Z-stack detected (Z = {img.shape[1]}): calculating maximum intensity projection")
        img = np.max(img, axis=0)
        n = 1
        c, h, w = img.shape
    else:
        n = 1
        c, h, w = img.shape
    
    # Preallocate resized array: flatten z and channel dims into one
    resized = np.empty((n * c, *(size, size)), dtype=img.dtype)
    
    # Resize each frame individually
    # Note: for now n = 1, but this may change later.
    for n_i in range(n):
        for c_i in range(c):
            index = n_i * c + c_i # global frame index
            if HF_settings.get("running_mode") in ["preprocess", "train"] and HF_ImageInfo.is_hidden_channel(c_i + 1):
                HF_log.info(f"Hiding channel {c_i + 1} in {HF_ImageInfo.get_name()}")
                frame = np.zeros((size, size), dtype=img.dtype)
            else:
                frame = img[c_i] if n == 1 else img[n_i][c_i]
            resized[index] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)

    ratio = size / w

    # Return channels as a 1-based index dict for external consistency.
    return {i + 1: resized[i] for i in range(n * c)}, ratio, (n, c)



def save_gray_as_rgb(channel, out_path, normalize=True):
    """
    Save a single-channel image as an RGB grayscale JPEG.

    This function takes a 2D array representing a single grayscale channel,
    optionally normalizes it to the [0, 1] range, replicates the values across
    three channels (R, G, B), converts the result to 8-bit unsigned integers,
    and saves it as a JPEG image.

    :param channel: 2D array of grayscale intensities (e.g. a channel from a multi-channel TIFF).
    :type channel: numpy.ndarray
    :param out_path: Destination path for the saved image.
    :type out_path: str
    :param normalize: Whether to normalize values to the [0, 1] range before conversion.
    :type normalize: bool, optional
    :rtype: None
    """
    gray = channel.astype(np.float32)
    if normalize:
        gray = (gray - gray.min()) / (gray.max() - gray.min())
    rgb_uint8 = (np.stack([gray, gray, gray], axis=-1) * 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(out_path, "JPEG")



def colorize_with_hue(frame, hue):
    """
    Apply a hue-based false coloring to a grayscale frame.

    The frame is normalized to [0, 1], then used as the V (value) channel in HSV,
    with S (saturation) fixed at 1 and H (hue) fixed to the given value.

    :param frame: 2D grayscale image.
    :type frame: np.ndarray
    :param hue: Hue value in [0, 1].
    :type hue: float
    :return: RGB image array of the same height/width as the input.
    :rtype: np.ndarray
    """
    # Normalize to [0, 1]
    norm = frame.astype(np.float32)
    norm /= norm.max() if norm.max() > 0 else 1

    # Build HSV channels
    h = np.full_like(norm, hue)  # constant hue
    s = np.ones_like(norm)       # full saturation
    v = norm                     # intensity from original frame

    hsv = np.stack([h, s, v], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb



def compose_hue_fusion(channels, selected_channels, palette, noise_channels=None):
    """
    Compose an RGB image by blending selected channels with hue-based coloring.

    Each selected channel is colorized using a hue from the given palette,
    then added to the RGB sum. Optional noise channels are colorized and
    added with reduced weight for visual texture.

    :param channels: Mapping of channel index to 2D grayscale frame.
    :type channels: dict[int, np.ndarray]
    :param selected_channels: Channels to render at full intensity.
    :type selected_channels: list[int]
    :param palette: Name of the color palette to use.
    :type palette: str
    :param noise_channels: Optional list of channels to render as 30%-intensity noise.
    :type noise_channels: list[int] | None
    :return: RGB image array in uint8 format, shape (H, W, 3), scaled to [0, 255].
    :rtype: np.ndarray
    """

    # Get height/width from the first channel
    h, w = next(iter(channels.values())).shape
    # Start with a black RGB canvas
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Render selected channels in full intensity.
    for ch in selected_channels:
        frame = channels[ch]
        hue = HF_palette.get_color(ch, palette=palette)[0]
        colored = colorize_with_hue(frame, hue) 
        rgb += colored

    # Render noise channels at reduced intensity for subtle visual hints.
    if noise_channels:
        for ch in noise_channels:
            frame = channels[ch]
            hue = HF_palette.get_color(ch, palette=palette)[0]
            noise = colorize_with_hue(frame, hue)
            rgb += 0.3 * noise 

    # Clamp values to [0, 1] to avoid saturation overflow.
    rgb = np.clip(rgb, 0, 1)
    # Scale to [0, 255] and convert to 8-bit integer
    return (rgb * 255).astype(np.uint8)

