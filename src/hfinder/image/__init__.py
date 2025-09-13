
# Image processing functions
from .processing import (
    normalize_channel,
    extract_frame,
    resize_multichannel_image,
    save_gray_as_rgb,
    colorize_with_hue,
    compose_hue_fusion
)


__all__ = [
    "normalize_channel",
    "extract_frame",
    "resize_multichannel_image",
    "save_gray_as_rgb",
    "colorize_with_hue",
    "compose_hue_fusion"
]
