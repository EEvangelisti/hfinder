""" 
Configuration manager for HFinder.

This module manages global configuration settings for the HFinder pipeline.

Overview
--------
- Load settings from a JSON file (`hfinder_settings.json`) into a global
  `SETTINGS` dictionary.
- Normalize types (via `pydoc.locate`) and defaults at import time.
- Provide helpers to:
  * gate CLI options by execution mode,
  * auto-populate an `argparse.ArgumentParser`,
  * merge parsed CLI args back into `SETTINGS`,
  * retrieve/update settings programmatically,
  * print a concise summary of effective values.

Public API
----------
- compatible_modes(actual, expected): Check if a parameter applies to a mode.
- define_arguments(parser, mode): Add CLI args from SETTINGS to an argparse parser.
- load(args): Apply parsed CLI arguments into SETTINGS (with type coercion).
- get(key): Retrieve the effective value for a setting (supports indirection).
- set(key, value, replace=True): Update/insert a setting value.
- print_summary(): Log the current effective settings via HFinder_log.

Notes
-----
- Each setting is defined by a *short name*, *long name*, *type*, *default*,
  and an optional *mode*. Types are provided as dotted names (e.g. "int",
  "float", "tuple", "str") and resolved to Python types. Booleans must be
  strings "true"/"false" in the JSON.
- If a setting has a "long" alias, `SETTINGS[<long>]` stores the *short* name,
  allowing indirection in lookups (see `get()`).
"""

import os
import sys
import ast
import json
import pydoc
import importlib.resources as ir
from hfinder.core import hfinder_log as HFinder_log
from hfinder.core import hfinder_utils as HFinder_utils
from glob import glob

    
# Each setting is defined by a short name, a long name, a type, a default value,
# and an optional mode. Types are given as Python dotted strings (e.g. `"int"`, 
# `"float"`, `"tuple"`, `"str"`), and converted automatically upon loading. 
# Booleans must be given as strings `"true"` / `"false"`.

SETTINGS = HFinder_utils.load_argument_list("hfinder_settings.json") or {}


# Normalize entries: resolve types, register long-name indirections, and
# coerce textual defaults into the proper Python types.
for key in list(SETTINGS.keys()):
    elt = SETTINGS[key]

    elt["type"] = pydoc.locate(elt["type"]) if "type" in elt else bool
    # Map long name to short key for indirection in get()/load().
    SETTINGS[elt["long"]] = key
    # Coerce string defaults to the declared Python type.
    if "default" in elt:
        elt["default"] = py_type(elt["default"])



def compatible_modes(actual, expected):
    """
    Determine if a parameter is relevant for a given execution mode.

    Used to dynamically include only the relevant CLI options for a script
    mode (e.g., filtering options for "train" vs "infer").

    :param actual: Mode attached to a parameter (e.g., "train", "test", or "*").
    :type actual: str
    :param expected: Mode requested for the current run.
    :type expected: str
    :return: True if the parameter applies to the expected mode (or is "*").
    :rtype: bool
    """
    return (actual == "*") \
        or (actual == expected) \
        or (expected in actual.split("|"))



def define_arguments(parser, mode):
    """
    Populate an argparse.ArgumentParser with CLI args defined in SETTINGS.

    Only arguments whose "mode" matches the provided *mode* (or is "*") are
    added. Each argument uses:
      - short and long flags (-x, --longname),
      - the declared type (int, float, str, etc.),
      - the default value (coerced at import),
      - a help string that includes the default value.

    :param parser: The argparse parser to augment.
    :type parser: argparse.ArgumentParser
    :param mode: Current operating mode used to filter arguments.
    :type mode: str
    :rtype: None
    """
  
    # Select only non-alias entries relevant to the requested mode.
    subset = {x: SETTINGS[x] for x in SETTINGS.keys() 
              if not isinstance(SETTINGS[x], str) and \
              compatible_modes(SETTINGS[x]["mode"], mode)}

    for cmd in subset.keys():
        param = SETTINGS[cmd]
        config = param["config"]
        if "default" in config:
            config["help"] = f"{config['help']} (default: {config['default']})"
        parser.add_argument(f"-{cmd}", f"--{param['long']}", **config)



def load(args):
    """
    Merge parsed CLI arguments back into SETTINGS (with safe coercion).

    For each key present in the argparse namespace:
      - Resolve any indirection (long → short).
      - Coerce strings to the declared type (including tuple/list via literal_eval).
      - Overwrite the corresponding "default" value in-place.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :rtype: None
    :raises ValueError: If a value cannot be parsed or converted to the expected type.
    """
    global SETTINGS
    user_defined = vars(args)

    for key, val in user_defined.items():
        if key in SETTINGS:
            # Resolve indirection (if SETTINGS[key] is a string, it points to the short key).
            if isinstance(SETTINGS[key], str):
                elt = SETTINGS[SETTINGS[key]]
            else:
                elt = SETTINGS[key]
            expected_type = elt["type"]
            
            # Parse textual tuples/lists (e.g., "(1,2,3)" or "[1,2,3]") if needed.
            if isinstance(val, str) and expected_type in (tuple, list):
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    raise ValueError(f"Could not parse {val} as {expected_type}")
            
            # Coerce to the expected type when necessary.
            if not isinstance(val, expected_type):
                try:
                    val = expected_type(val)
                except Exception:
                    raise ValueError(f"Could not convert {val} to {expected_type}")

            # Update the effective default for downstream consumers.
            elt["config"]["default"] = val



def get(key):
    """
    Retrieve an effective setting value by key (supports long/short indirection).

    If *key* maps to a string in SETTINGS, it is treated as an alias pointing
    to the canonical short-name entry; otherwise, the value is read directly.

    :param key: Setting name (short or long).
    :type key: str
    :return: The effective value, or None if the key is unknown.
    :rtype: Any | None
    """
    if key in SETTINGS:
        out = SETTINGS[key]
        if isinstance(out, str):
            # Indirection: `SETTINGS[long] = short`
            if out in SETTINGS:
                return SETTINGS[out]["config"]["default"]
            else:
                # Fallback: if alias does not resolve, return associated value.
                return out
        else:
            return out["config"]["default"]
    else:
        return None



def set(key, value, replace=True):
    """
    Insert or update a setting entry in SETTINGS.

    If *key* exists and *replace* is True, the entry is overwritten. Otherwise,
    a new key is inserted.

    :param key: Setting name to set.
    :type key: str
    :param value: Value to assign.
    :type value: Any
    :param replace: Whether to overwrite existing entries.
    :type replace: bool
    :rtype: None
    """
    global SETTINGS
    if key in SETTINGS:
        if replace:
            SETTINGS[key] = value
    else:
        SETTINGS[key] = value



def print_summary():
    """
    Log a summary of effective settings (long name → default value).

    Only dictionary entries (canonical settings) are reported; alias strings
    are ignored for clarity.

    :rtype: None
    """
    global SETTINGS
    for key in SETTINGS.keys():
        if isinstance(SETTINGS[key], dict):
            if compatible_modes(SETTINGS[key]['mode'], SETTINGS["running_mode"]):
                value = False if "default" not in SETTINGS[key]["config"] else SETTINGS[key]["config"]['default']
                HFinder_log.info(f"[SETTING] '{SETTINGS[key]['long']}' = {value}")



def load_class_definitions():
    """
    Extract class names from JSON files in `tiff_dir/classes/` and assign IDs.

    Class names are inferred from file basenames, sorted alphabetically, and
    assigned increasing integer IDs starting at 0.

    :return: Mapping from class name to class ID, e.g., {"hyphae": 0, "nuclei": 1}.
    :rtype: dict[str, int]
    """
    class_dir = os.path.join(get("tiff_dir"), "hf_classes")
    if not os.path.isdir(class_dir):
        HFinder_log.fail(f"No such directory: {class_dir}")

    files = sorted(glob(os.path.join(class_dir, "*.json")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return {name: i for i, name in enumerate(names)}



