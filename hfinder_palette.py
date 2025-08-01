import hashlib
import colorsys
import hfinder_log as HFinder_log

base_palette_rgb = [
    (255, 0, 255),   # magenta
    (0, 255, 255),   # jaune clair
    (0, 255, 0),     # vert
    (255, 0, 0),     # bleu
    (0, 128, 255),   # orange
    (128, 0, 255),   # violet
    (255, 255, 0),   # cyan
    (255, 128, 0),   # corail
]

def get_color(n, palette=base_palette_rgb):
    return palette[n % len(palette)]

# Normalise et convertit en HSV
base_palette_hsv = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in base_palette_rgb]

def rotated_palette_hsv(delta_hue=0.0):
    return [((h + delta_hue) % 1.0, s, v) for h, s, v in base_palette_hsv]
    
def rotated_palette_rgb(delta_hue=0.0):
    rotated_hsv = rotated_palette_hsv(delta_hue)
    return [tuple(int(255 * x) for x in colorsys.hsv_to_rgb(h, s, v)) for h, s, v in rotated_hsv]

def get_random_palette(colorspace="RGB", hash_data=None):
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
    if colorspace.upper() == "RGB":
        return rotated_palette_rgb(delta)
    elif colorspace.upper() == "HSV":
        return rotated_palette_hsv(delta)
    else:
        HFinder_log.fail(f"Unknown colorspace {colorspace}")
