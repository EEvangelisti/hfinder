import os
from datetime import datetime
import hfinder_log as HFinder_log

def get_timestamp():
    """
    Returns the current date and time as a string formatted as 'YYYY-MM-DD_HH-MM-SS'.
    This format is suitable for creating unique, chronologically ordered folder or file names.
    Returns:
        str: The current timestamp, e.g., '2025-07-27_18-24-03'.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_tree(base, tree):
    """
    Recursively creates a directory structure from a nested dictionary.
    Each key in the dictionary represents a folder name. If the value is another 
    dictionary, the function recurses into it. If the value is not a dictionary 
    (e.g., a list, None, etc.), the folder is created but not recursed further.

    Args:
        base (str): The root path where the directory tree should be created.
        tree (dict): A nested dictionary representing the folder hierarchy.

    Example:
        create_tree("2025-07-27_18-42-00", {
            "dataset": {
                "images": {"train": [], "val": []},
                "labels": {"train": [], "val": []}
            },
            "log": []
        })
        """
    os.makedirs(base, exist_ok=True)
    if isinstance(tree, dict):
        for name, subtree in tree.items():
            path = os.path.join(base, name)
            create_tree(path, subtree)       


def create_training_folders():
    """
    Creates a timestamped root directory containing a standard dataset structure 
    for machine learning training and validation, and returns the full tree.

    The structure is as follows:
        <timestamp>/
        ├── dataset/
        │   ├── masks/
        │   ├── contours/
        │   ├── images/
        │   │   ├── train/
        │   │   └── val/
        │   ├── labels/
        │   │   ├── train/
        │   │   └── val/
        └── log/

    Returns:
        dict: A nested dictionary representing the folder structure with an additional
              "root" key pointing to the absolute path of the created base directory.

    Example:
        tree = create_training_folders()
        print(tree["root"])  # e.g., /home/user/2025-07-27_18-52-11
    """
    HFinder_log.info("Creating folders")
    base = get_timestamp()
    folder_tree = {
        "dataset": {
            "masks": [],
            "contours": [],
            "images": {
                "train": [],
                "val": [],
            },
            "labels": {
                "train": [],
                "val": [],
            },
        },
        "log": []
    }
    create_tree(base, folder_tree)
    folder_tree["root"] = os.path.abspath(base)
    return folder_tree


def get_subtree(tree, path):
    """
    Retrieves a nested subtree from a dictionary-based tree structure,
    following a '/'-separated path.

    Parameters
    ----------
    tree : dict
        The nested dictionary representing the tree.
    path : str
        The path to the desired subtree, e.g., "dataset/images/train".

    Returns
    -------
    dict or list
        The subtree located at the given path.

    Raises
    ------
    KeyError
        If the path does not exist in the tree.
    TypeError
        If a non-dict is encountered before the final node.
    """
    current = tree
    for key in path.strip("/").split("/"):
        if not isinstance(current, dict):
            raise TypeError(f"Expected dict at {key}, but got {type(current)}")
        current = current[key]
    return current    


def set_subtree(tree, path, value):
    """
    Sets a nested subtree in a dictionary-based tree structure,
    following a '/'-separated path. Intermediate dictionaries are created
    if they do not exist.

    Parameters
    ----------
    tree : dict
        The root dictionary representing the tree.
    path : str
        The path to set, e.g., "dataset/images/train".
    value : any
        The value to assign at the target path (e.g., list, dict, etc.).

    Raises
    ------
    TypeError
        If an intermediate path component exists but is not a dict.
    """
    keys = path.strip("/").split("/")
    current = tree
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise TypeError(f"Cannot assign subtree: {key} is not a dict.")
        current = current[key]
    current[keys[-1]] = value
    
    
def append_subtree(tree, path, value):
    """
    Append a value to a list located at a given path within a nested dictionary tree structure.

    The path is a string with keys separated by slashes (e.g., "a/b/c"). Intermediate subdictionaries
    are created as needed. The function raises an error if any intermediate node exists but is not
    a dictionary, or if the final target is not a list.

    Args:
        tree (dict): The nested dictionary to be updated.
        path (str): A slash-separated string representing the path to the target list.
        value (Any): The value to append to the list at the target location.

    Raises:
        TypeError: If any part of the path corresponds to a non-dictionary object,
                   or if the final key does not point to a list.
    """
    keys = path.strip("/").split("/")
    current = tree
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise TypeError(f"(HFinder) Cannot assign subtree: '{key}' is not a dict.")
        current = current[key]

    final_key = keys[-1]
    if final_key not in current:
        current[final_key] = []
    elif not isinstance(current[final_key], list):
        raise TypeError(f"(HFinder) Cannot append to '{final_key}': not a list.")
    current[final_key].append(value)


def get_image_train_dir():
    return os.path.join("dataset", "images", "train")
    
def get_label_train_dir():
    return os.path.join("dataset", "labels", "train")   

def get_image_val_dir():
    return os.path.join("dataset", "images", "val")

def get_label_val_dir():
    return os.path.join("dataset", "labels", "val")   

     
