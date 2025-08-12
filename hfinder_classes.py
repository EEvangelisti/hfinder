import os
import json
from glob import glob
import hfinder_log as HFinder_log
import hfinder_settings as HFinder_settings


CLASS_INSTRUCTIONS = None


def initialize():
    """
    Load and consolidate image-to-class channel mappings from JSON annotation files.

    This function reads all JSON files in the `classes/` subdirectory of the 
    TIFF data directory (as specified in the HFinder settings). Each file must 
    be named after a class and contain a mapping from image filenames to either:
      - an integer (representing the channel for that class in the image), or
      - a dictionary containing at least the key `"channel"`.

    The function first builds per-class mappings, then merges them into a 
    unified image-wise map where, for each image, every class is listed with 
    its associated channel mapping or `-1` if absent.

    Returns:
        dict: A nested dictionary of the form:
              {
                  "image_name_1": {
                      "class_A": {"channel": X},
                      "class_B": -1,
                      ...
                  },
                  ...
              }

    Raises:
        - Logs a fatal error and exits if:
          - The `classes/` directory does not exist.
          - Any annotation is malformed (missing `"channel"` key or incorrect format).

    """
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



def has_image(image_file):
    global CLASS_INSTRUCTIONS
    assert CLASS_INSTRUCTIONS is not None, "(HFinder) Assert Failure: has_image"
    return image_file in CLASS_INSTRUCTIONS



def from_frame(image_file, default=0):
    global CLASS_INSTRUCTIONS
    assert CLASS_INSTRUCTIONS is not None, "(HFinder) Assert Failure: from_frame"
    if image_file in CLASS_INSTRUCTIONS:
        if "from" in CLASS_INSTRUCTIONS[image_file]:
            return CLASS_INSTRUCTIONS[image_file]["from"] - 1
        else:
            return default
    else:
        return default


# !!! Missing class
def get_items(image_file):
    global CLASS_INSTRUCTIONS
    assert CLASS_INSTRUCTIONS is not None, "(HFinder) Assert Failure: get_items"    
    if image_file in CLASS_INSTRUCTIONS:
        return CLASS_INSTRUCTIONS[image_file].items()
    else:
        return {}.items()


# !!! Missing class
def to_frame(image_file, default=0):
    global CLASS_INSTRUCTIONS
    assert CLASS_INSTRUCTIONS is not None, "(HFinder) Assert Failure: to_frame"
    if image_file in CLASS_INSTRUCTIONS:
        if "to" in CLASS_INSTRUCTIONS[image_file]:
            return CLASS_INSTRUCTIONS[image_file]["to"] - 1
        else:
            return default
    else:
        return default


def allows_MIP_generation(image_file, class_name):
    global CLASS_INSTRUCTIONS
    assert CLASS_INSTRUCTIONS is not None, "(HFinder) Assert Failure: allows_MIP_generation"
    try:
        CLASS_INSTRUCTIONS[image_file][class_name]["MIP"] > 0
    except:
        return True






