""" 
Color palette rotation and generation utilities for HFinder.

Overview
--------
- Build a base HSV palette by evenly spacing hues on the circle.
- Retrieve a color by index with wrap-around.
- Create hue-rotated variants of the base palette.
- Generate a deterministic "random" palette from a hashable token.

Public API
----------
- get_color(n, palette=base_palette_hsv): Fetch the nth color (wrap-around).
- rotated_palette_hsv(delta_hue=0.0): Rotate all hues by `delta_hue`.
- get_random_palette(hash_data=None): Rotate palette using a random or hashed shift.

Notes
-----
- All colors are represented as HSV tuples with values in [0, 1].
- `get_random_palette` returns an HSV palette (same format as the base palette).
"""

import hashlib
import colorsys
import numpy as np
import hfinder.core.hfinder_log as HFinder_log



# ---------------------------------------------------------------------
# Base palette construction (HSV)

base_palette_hsv = []

N = 6
start_hex="#FF0000"
# # Convert hex to RGB in [0,1]
start_rgb = tuple(int(start_hex[i:i+2], 16)/255 for i in (1, 3, 5))
start_hsv = colorsys.rgb_to_hsv(*start_rgb)

# Generate N hues by spacing hue evenly around the circle
for i in range(N):
    h = (start_hsv[0] + i / N) % 1.0  # hue ∈ [0,1]
    s = start_hsv[1]
    v = start_hsv[2]
    base_palette_hsv.append((h, s, v))
# ---------------------------------------------------------------------



def get_color(n, palette=base_palette_hsv):
    """
    Return the color at index `n` from `palette`, cycling if needed.

    :param n: Index of the desired color (wrap-around if n >= len(palette)).
    :type n: int
    :param palette: Sequence of HSV tuples (values ∈ [0,1]).
    :type palette: list[tuple[float, float, float]]
    :return: HSV color tuple (h, s, v), each in [0,1].
    :rtype: tuple[float, float, float]
    """
    return palette[n % len(palette)]



def rotated_palette_hsv(delta_hue=0.0):
    """
    Produce a new HSV palette by rotating the hue of each base color.

    The rotation is applied modulo 1.0 (hue wraps around the unit circle).

    :param delta_hue: Hue rotation amount in [0.0, 1.0].
    :type delta_hue: float
    :return: Rotated palette as HSV tuples (h, s, v).
    :rtype: list[tuple[float, float, float]]
    """
    return [((h + delta_hue) % 1.0, s, v) for h, s, v in base_palette_hsv]



def get_random_palette(hash_data=None):
    """
    Generate a hue-rotated palette using a random or deterministic shift.

    If `hash_data` is provided, a stable rotation is derived from its SHA-256
    digest; otherwise a fresh random rotation in [0,1) is used.

    :param hash_data: Optional token (str or bytes) used to derive a deterministic shift.
    :type hash_data: str | bytes | None
    :return: HSV palette (list of (h, s, v) tuples) after rotation.
    :rtype: list[tuple[float, float, float]]
    """
    if hash_data is None:
        delta = np.random.rand()
    else:
        # Convert to bytes then hash with SHA-256
        if isinstance(hash_data, str):
            hash_data = hash_data.encode('utf-8')
        hash_bytes = hashlib.sha256(hash_data).digest()
        # Map first 4 bytes to a float in [0,1) with ~1e-6 resolution
        hash_int = int.from_bytes(hash_bytes[:4], 'big')
        delta = (hash_int % 10**6) / 10**6

    return rotated_palette_hsv(delta)

