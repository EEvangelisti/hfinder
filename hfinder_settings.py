""" 
hfinder_settings — Configuration manager

This module manages global configuration settings for the HFinder pipeline.

Default configuration:
- `"tiff_dir"`: path to the directory containing input TIFF files (default: `"data"`).
- `"default_auto_threshold"`: default automatic threshold used for binarization (default: `90`).
- `"target_size"`: target image size (width, height) for resizing before processing (default: `(640, 640)`).
- `"epochs"`: number of training epochs for the YOLO model (default: `100`).
- `"model"`: YOLOv8 segmentation model name or path (default: `"yolov8n-seg.pt"`).
"""

SETTINGS = {
    "tiff_dir": "data",
    "default_auto_threshold": 90,
    "target_size" : (640, 640),
    "epochs": 100,
    "model": "yolov8n-seg.pt",
    "mode": "normal",
    "validation_frac": 0.2
}

def load(args):
    """
    Update the global SETTINGS dictionary using values from argparse arguments.

    Parameters:
        args (argparse.Namespace): The parsed command-line arguments.
    
    Note:
        Existing keys in SETTINGS will be overwritten if present in args.
    """
    global SETTINGS
    SETTINGS = {**SETTINGS, **vars(args)}



def get(key):
    """
    Retrieve a value from the SETTINGS dictionary by key.

    Parameters:
        key (str): The name of the setting.

    Returns:
        The corresponding value if found, or None otherwise.
    """
    return SETTINGS[key] if key in SETTINGS else None
