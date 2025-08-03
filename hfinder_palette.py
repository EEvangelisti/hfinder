""" Color Palette Rotation and Generation Utilities

This module provides utilities to handle and dynamically rotate a predefined 
color palette using the RGB and HSV color spaces. It is particularly useful when
visual distinction between colors is essential (e.g., in image processing or 
segmentation tasks).
"""

import hashlib
import colorsys
import hfinder_log as HFinder_log

base_palette_hsv = []

N = 6
start_hex="#FF0000"
# Convert hex to RGB in [0,1]
start_rgb = tuple(int(start_hex[i:i+2], 16)/255 for i in (1, 3, 5))
start_hsv = colorsys.rgb_to_hsv(*start_rgb)

# Generate N hues by spacing hue evenly around the circle
for i in range(N):
    h = (start_hsv[0] + i / N) % 1.0  # hue ∈ [0,1]
    s = start_hsv[1]
    v = start_hsv[2]
    base_palette_hsv.append((h, s, v))



def get_color(n, palette=base_palette_hsv):
    """
    Returns the color at position n in the given palette, cycling back to the 
    start if n exceeds the palette length.

    Arguments
    n (int): Index of the desired color.
    palette (list[tuple]): Optional. A list of RGB/HSV tuples. 
    Defaults to base_palette_rgb.

    Returns
    tuple[int, int, int]: An HSV color.
    """
    return palette[n % len(palette)]



def rotated_palette_hsv(delta_hue=0.0):
    """
    Returns a new HSV palette by rotating the hue of each base color by 
    delta_hue. The resulting hues are wrapped around the unit circle (mod 1.0).

    Arguments
    delta_hue (float): Amount to rotate the hue, in the range [0.0, 1.0].

    Returns
    list[tuple[float, float, float]]: HSV tuples after rotation.
    """
    return [((h + delta_hue) % 1.0, s, v) for h, s, v in base_palette_hsv]



def get_random_palette(hash_data=None):
    """
    Generates a rotated version of the base palette, with a hue shift determined
    either randomly or from a hash of user-provided data (e.g. a filename or ID).

    Arguments
    colorspace (str): Either "RGB" or "HSV". Determines the format of the returned palette.
    hash_data (str | bytes | None): If None, a random hue shift is applied. 
    Otherwise, a deterministic value is computed from the SHA256 hash of the input.

    Returns
    list[tuple]: Rotated palette in RGB or HSV format.

    Exceptions
    Raises a runtime error through HFinder_log.fail() if the colorspace is unrecognized.
    """
    if hash_data is None:
        delta = np.random.rand()
    else:
        # Conversion to bytes, then to hash SHA256
        if isinstance(hash_data, str):
            hash_data = hash_data.encode('utf-8')
        hash_bytes = hashlib.sha256(hash_data).digest()
        # Create a float between 0 and 1 for the first four bits
        hash_int = int.from_bytes(hash_bytes[:4], 'big')
        delta = (hash_int % 10**6) / 10**6  # Resolution of 1e-6

    return rotated_palette_hsv(delta)

