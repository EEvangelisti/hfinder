import os
import json
from glob import glob
import hfinder_settings as HFinder_settings



def load_class_definitions():
    """
    Extract class names from all JSON files in `data_dir/classes/`, assign them integer IDs.

    Returns:
        dict: Mapping from class name to class ID (int), e.g., {"hyphae": 0, "nuclei": 1}
    """
    class_dir = os.path.join(HFinder_settings.get("tiff_dir"), "classes")
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"No such directory: {class_dir}")

    class_files = sorted(glob(os.path.join(class_dir, "*.json")))
    class_names = [os.path.splitext(os.path.basename(f))[0] for f in class_files]
    return {name: idx for idx, name in enumerate(class_names)}





