#
#

SETTINGS = {
    "tiff_dir": "data",
    "channels": "hyphae.json",
    "thresholds": "thresholds.json",
    "target_size" : (640, 640),
    "epochs": 100,
}

def load(args):
    global SETTINGS
    args_dict = vars(args)
    SETTINGS = {**SETTINGS, **vars(args)}

def get(key):
    return SETTINGS[key] if key in SETTINGS else None
