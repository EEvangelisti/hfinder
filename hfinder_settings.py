""" 
hfinder_settings — Configuration manager

This module manages global configuration settings for the HFinder pipeline.
"""

import ast
import json
from pydoc import locate
import hfinder_log as HFinder_log

# Each setting is defined by a short name, a long name, a type, a default value,
# and an optional mode. Types are given as Python dotted strings (e.g. `"int"`, 
# `"float"`, `"tuple"`, `"str"`), and converted automatically upon loading. 
# Booleans must be given as strings `"true"` / `"false"`.
SETTINGS = {}

with open("hfinder_settings.json", "r") as f:
    SETTINGS = json.load(f)

for key in list(SETTINGS.keys()):
    elt = SETTINGS[key]

    py_type = locate(elt["type"])
    elt["type"] = py_type
    if "long" in elt:
        SETTINGS[elt["long"]] = key

    if elt["type"] is tuple:
        elt["default"] = ast.literal_eval(elt["default"])
    else:
        elt["default"] = py_type(elt["default"])



def compatible_modes(actual, expected):
    """
    Checks whether a command-line argument should be included based on its 
    applicable mode. Used to dynamically include only the relevant arguments 
    for a given script mode (e.g., filtering CLI options for training vs. 
    predicting).

    Parameters:
        actual (str): The mode assigned to a parameter (e.g., "train", "test", 
        or "*" for universal use).
        expected (str): The mode currently requested (e.g., the current 
        operation mode of the script).

    Returns:
    bool: True if the argument is relevant to the expected mode, or if it is 
    universal ("*") — otherwise False.
    """
    return actual == "*" or actual == expected



def define_arguments(parser, mode):
    """
    Automatically populates a given argparse.ArgumentParser with arguments 
    defined in a centralized SETTINGS JSON object. Only parameters matching the 
    given mode (or marked with "*" for universal use) are included.

    Parameters:
        parser (argparse.ArgumentParser): The parser instance to which arguments
        will be added.
        mode (str): The current operating mode, used to filter relevant
        arguments (e.g., "train", "infer").

    Behavior:
        For each entry in the global SETTINGS dictionary:
        If the mode matches (or is "*"), it adds a command-line argument with:
            - short and long flags (-x, --longname)
            - the appropriate type (e.g., int, float, str)
            - the default value
            - a human-readable help string including the default
    """
    subset = {x: SETTINGS[x] for x in SETTINGS.keys() 
              if not isinstance(SETTINGS[x], str) and \
              compatible_modes(SETTINGS[x]["mode"], mode)}

    for cmd in subset.keys():
        parser.add_argument(f"-{cmd}", f"--{SETTINGS[cmd]['long']}",
                            type=SETTINGS[cmd]["type"],
                            default=SETTINGS[cmd]["default"],
                            help=f"{SETTINGS[cmd]['help']} (default: \
                                 {SETTINGS[cmd]['default']})")



def load(args):
    """
    Update the global SETTINGS dictionary using values from argparse arguments.

    Parameters:
        args (argparse.Namespace): The parsed command-line arguments.
    
    Note:
        Existing keys in SETTINGS will be overwritten if present in args.
    """
    global SETTINGS
    user_defined = vars(args)

    for key, val in user_defined.items():
        if key in SETTINGS:
            if isinstance(SETTINGS[key], str):
                elt = SETTINGS[SETTINGS[key]]
            else:
                elt = SETTINGS[key]
            expected_type = elt["type"]
            
            if isinstance(val, str) and expected_type in (tuple, list):
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    raise ValueError(f"Could not parse {val} as {expected_type}")
            
            if not isinstance(val, expected_type):
                try:
                    val = expected_type(val)
                except Exception:
                    raise ValueError(f"Could not convert {val} to {expected_type}")

            elt["default"] = val



def get(key):
    """
    Retrieve a value from the SETTINGS dictionary by key.

    Parameters:
        key (str): The name of the setting.

    Returns:
        The corresponding value if found, or None otherwise.
    """
    if key in SETTINGS:
        if isinstance(SETTINGS[key], str):
            return SETTINGS[SETTINGS[key]]["default"]
        else:
            return SETTINGS[key]["default"]
    else:
        return None
        


def print_summary():
    global SETTINGS
    for key in SETTINGS.keys():
        if isinstance(SETTINGS[key], dict):
            HFinder_log.info(f"{SETTINGS[key]['long']}: {SETTINGS[key]['default']}")



        
        
