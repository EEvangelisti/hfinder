#
#

SETTINGS = {
    "tiff_dir": "data",
    "default_auto_threshold": 90,
    "target_size" : (640, 640),
    "epochs": 100,
    "model": "yolov8n-seg.pt"
}

def load(args):
    global SETTINGS
    args_dict = vars(args)
    SETTINGS = {**SETTINGS, **vars(args)}

def get(key):
    return SETTINGS[key] if key in SETTINGS else None
