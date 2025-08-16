"""
Image/class instruction resolver for HFinder.

This module loads per-class JSON annotations (one JSON per class) from
`tiff_dir/classes/` and merges them into an image-wise instruction mapping.
Each image then exposes, for each class, either a dictionary of directives
(e.g. channel, threshold, frame bounds, segmentation path, MIP flag) or
the sentinel value -1 when no directives exist for that class.

Public API
----------
- initialize(): Populate the global CLASS_INSTRUCTIONS from JSON files.
- set_current_image(img_name): Select the active image.
- image_has_instructions(): Tell whether the current image is known.
- get_classes(): Enumerate classes with valid instructions for the current image.
- Accessors (per class on the current image):
  - get_current(cls)             -> dict | int
  - from_frame(cls, default=0)   -> int (0-based)
  - to_frame(cls, default=0)     -> int (0-based)
  - get_channel(cls)             -> int | None
  - get_threshold(cls)           -> float | int | None
  - get_manual_segmentation(cls) -> str | None
  - allows_MIP_generation(cls)   -> bool

Notes
-----
- Frame indices are stored 1-based in JSON and converted to 0-based here.
- JSON values may be either:
  * an integer channel index, or
  * a dict containing at least the key "channel", plus optional keys:
    "from", "to", "threshold", "segment", "MIP".
- Missing/invalid fields are handled defensively and fall back to defaults.
"""

import os
import json
from glob import glob
import hfinder_log as HFinder_log
import hfinder_settings as HFinder_settings

# Name of the image currently being processed by the pipeline.
CURRENT_IMAGE = None
# Class currently being processed by the pipeline.
CURRENT_CLASS = None

# Master mapping built by initialize():
#   CLASS_INSTRUCTIONS[image_name][class_name] = dict(...) | -1
# Set to None until initialize() has successfully completed.
CLASS_INSTRUCTIONS = None



def initialize():
    """
    Discover and load per-class JSON annotations, then build the
    image-wise instruction table.

    The function expects a directory structure like:
        tiff_dir/
          classes/
            classA.json
            classB.json
            ...

    Each JSON file maps image names to either:
      - an integer channel index (shorthand), or
      - an instruction dict containing at minimum {"channel": <int>}.
        Optional keys include "from", "to", "threshold", "segment", "MIP".

    On success, the global CLASS_INSTRUCTIONS takes the form:
        {
          "img1.tif": {
            "classA": {"channel": 0, ...},
            "classB": -1,
            ...
          },
          ...
        }

    Any malformed annotation triggers a logged failure with a specific exit code.

    :rtype: None
    """
    
    # Locate the classes directory from settings and validate it.
    tiff_dir = HFinder_settings.get("tiff_dir")
    class_dir = os.path.join(tiff_dir, "classes")
    if not os.path.isdir(class_dir):
        HFinder_log.fail(f"No such directory: {class_dir}", exit_code=4)

    # Gather all class definition files and derive class names from filenames.
    class_files = sorted(glob(os.path.join(class_dir, "*.json")))
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in class_files]

    # ----------------------------------------------------------------------
    # Step 1: build class-wise image mappings
    # For each class, ensure that every image maps to a normalized dict
    # containing at least {"channel": <int>}. A bare integer is accepted
    # as a shorthand for {"channel": <int>}.
    # ----------------------------------------------------------------------
    per_class_maps = {}
    for i, json_file in enumerate(class_files):
        class_name = class_names[i]
        with open(json_file, "r") as f:
            raw_map = json.load(f)
        per_class_maps[class_name] = {}
        for img, val in raw_map.items():
            if isinstance(val, int):
                # Shorthand: an int means "channel" only.
                per_class_maps[class_name][img] = {"channel": val}
            elif isinstance(val, dict) and "channel" in val:
                # Full-form annotation; keep as-is.
                per_class_maps[class_name][img] = val
            else:
                # Invalid structure for this image/class.
                HFinder_log.fail(f"Invalid annotation for image {img} in class {class_name}",
                                 exit_code=HFinder_log.EXIT_INVALID_ANNOTATION)

    # ----------------------------------------------------------------------
    # Step 2: merge into an image-wise mapping
    # Collect the union of all images mentioned across classes and, for each
    # image, emit a dict over all classes, using -1 when a class provides
    # no directives for that image.
    # ----------------------------------------------------------------------
    all_images = set()
    for d in per_class_maps.values():
        all_images.update(d.keys())

    final_map = {}
    for img in sorted(all_images):
        final_map[img] = {}
        for class_name in class_names:
            if img in per_class_maps[class_name]:
                final_map[img][class_name] = per_class_maps[class_name][img]
            else:
                final_map[img][class_name] = -1
                
    # Publish the merged lookup as a module-level singleton.
    global CLASS_INSTRUCTIONS
    CLASS_INSTRUCTIONS = final_map



def set_current_image(img_name):
    """
    Set the currently active image and log the operation.

    :param img_name: Name of the image to process.
    :type img_name: str
    :rtype: None
    """
    global CURRENT_IMAGE
    CURRENT_IMAGE = img_name
    HFinder_log.info(f"Processing {img_name}")


def set_current_class(cls_name):
    """
    Set the currently active class.

    :param img_name: Name of the image to process.
    :type img_name: str
    :rtype: None
    """
    global CURRENT_CLASS
    CURRENT_CLASS = cls_name



def get_current(img_class=None):
    """
    Retrieve the instruction dictionary for a given class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: dict | int
    """
    cls = CURRENT_CLASS if img_class is None else img_class
    return CLASS_INSTRUCTIONS[CURRENT_IMAGE][cls]



def image_has_instructions():
    """
    Check if the current image has associated instructions.

    :rtype: bool
    """
    return CURRENT_IMAGE in CLASS_INSTRUCTIONS



def from_frame(img_class=None, default=0):
    """
    Get the starting frame index for the current class, zero-based.

    :param cls: Class name.
    :type cls: str
    :param default: Value returned if no valid frame is found.
    :type default: int
    :rtype: int
    """
    try:
        cls = CURRENT_CLASS if img_class is None else img_class
        # Frames are 1-based in the JSON; return 0-based here.
        x = int(get_current(cls)["from"])
        return default if x <= 0 else x - 1
    except:
        return default



def to_frame(img_class=None, default=0):
    """
    Get the ending frame index for the current class, zero-based.

    :param cls: Class name.
    :type cls: str
    :param default: Value returned if no valid frame is found.
    :type default: int
    :rtype: int
    """
    try:
        cls = CURRENT_CLASS if img_class is None else img_class
        # Frames are 1-based in the JSON; return 0-based here.
        x = int(get_current(cls)["to"])
        return default if x <= 0 else x - 1
    except:
        return default



def get_classes():
    """
    List the classes that have valid instructions for the current image.

    :rtype: list[str]
    """
    try:
        classes = CLASS_INSTRUCTIONS[CURRENT_IMAGE].keys()
        # Filter out classes with the sentinel -1 (no directives for this image).
        return [cls for cls in classes if get_current(cls) != -1]
    except:
        return []


def get_channel(img_class=None):
    """
    Get the channel index for the current class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: int | None
    """
    try:
        cls = CURRENT_CLASS if img_class is None else img_class
        return get_current(cls)["channel"]
    except:
        return None



def get(key, img_class=None):
    """
    Get the value associated with `key` for the current class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: <multiple> | None
    """
    try:
        cls = CURRENT_CLASS if img_class is None else img_class
        return get_current(cls)[key]
    except:
        return None


def get_threshold(img_class=None):
    """
    Get the threshold value for the current class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: float | int | None
    """
    try:
        cls = CURRENT_CLASS if img_class is None else img_class
        return get_current(cls)["threshold"]
    except:
        return None


def get_manual_segmentation(img_class=None):
    """
    Get the filename of the manual segmentation for the current class.

    :param cls: Class name.
    :type cls: str
    :rtype: str | None
    """
    try:
        cls = CURRENT_CLASS if img_class is None else img_class
        return get_current(cls)["segment"]
    except:
        return None


def allows_MIP_generation(img_class=None):
    """
    Check whether MIP (Maximum Intensity Projection) generation is allowed
    for the current class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: bool
    """
    try:
        cls = CURRENT_CLASS if img_class is None else img_class
        return get_current(cls)["MIP"] > 0
    except:
        # Default to permissive when the flag is absent.
        return True


