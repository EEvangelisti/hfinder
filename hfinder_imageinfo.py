import os
import json
from glob import glob
import hfinder_log as HFinder_log
import hfinder_settings as HFinder_settings

CURRENT_IMAGE = None
CLASS_INSTRUCTIONS = None



def initialize():
    class_dir = os.path.join(HFinder_settings.get("tiff_dir"), "classes")
    if not os.path.isdir(class_dir):
        HFinder_log.fail(f"No such directory: {class_dir}", exit_code=4)

    class_files = sorted(glob(os.path.join(class_dir, "*.json")))
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in class_files]

    # Step 1: build class-wise image mappings
    per_class_maps = {}
    for json_file in class_files:
        class_name = os.path.splitext(os.path.basename(json_file))[0]
        with open(json_file, "r") as f:
            raw_map = json.load(f)
        per_class_maps[class_name] = {}
        for img, val in raw_map.items():
            if isinstance(val, int):
                per_class_maps[class_name][img] = {"channel": val}
            elif isinstance(val, dict) and "channel" in val:
                per_class_maps[class_name][img] = val
            else:
                HFinder_log.fail(f"Invalid annotation for image {img} in class {class_name}",
                                 exit_code=HFinder_log.EXIT_INVALID_ANNOTATION)

    # Step 2: merge into image-wise mapping
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



def get_current(cls):
    """
    Retrieve the instruction dictionary for a given class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: dict | int
    """
    return CLASS_INSTRUCTIONS[CURRENT_IMAGE][cls]



def image_has_instructions():
    """
    Check if the current image has associated instructions.

    :rtype: bool
    """
    return CURRENT_IMAGE in CLASS_INSTRUCTIONS



def from_frame(cls, default=0):
    """
    Get the starting frame index for a class, zero-based.

    :param cls: Class name.
    :type cls: str
    :param default: Value returned if no valid frame is found.
    :type default: int
    :rtype: int
    """
    try:
        x = int(get_current(cls)["from"])
        return default if x <= 0 else x - 1
    except:
        return default



def to_frame(cls, default=0):
    """
    Get the ending frame index for a class, zero-based.

    :param cls: Class name.
    :type cls: str
    :param default: Value returned if no valid frame is found.
    :type default: int
    :rtype: int
    """
    try:
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
        return [cls for cls in classes if get_current(cls) != -1]
    except:
        return []


def get_channel(cls):
    """
    Get the channel index for a given class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: int | None
    """
    try:
        return get_current(cls)["channel"]
    except:
        return None


def get_threshold(cls):
    """
    Get the threshold value for a given class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: float | int | None
    """
    try:
        return get_current(cls)["threshold"]
    except:
        return None


def get_manual_segmentation(cls):
    """
    Get the filename of the manual segmentation for a given class.

    :param cls: Class name.
    :type cls: str
    :rtype: str | None
    """
    try:
        return get_current(cls)["segment"]
    except:
        return None


def allows_MIP_generation(cls):
    """
    Check whether MIP (Maximum Intensity Projection) generation is allowed
    for a given class in the current image.

    :param cls: Class name.
    :type cls: str
    :rtype: bool
    """
    try:
        return get_current(cls)["MIP"] > 0
    except:
        return True


