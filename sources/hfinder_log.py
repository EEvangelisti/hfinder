"""
Logging and exit utilities for HFinder.

This module centralizes:
- Verbosity control for info/warning/error messages.
- Standardized message formatting with the "(HFinder)" prefix.
- Named exit codes for common error conditions.
- Helper to terminate execution immediately with an error code.
- Helper to format assertion failure messages.

Public API
----------
- set_verbosity(n): Set verbosity level (1=errors only, 2=errors+warnings, 
    3=errors+warnings+info).
- info(msg): Print an info message if verbosity >= 3.
- warn(msg): Print a warning message if verbosity >= 2.
- fail(msg, exit_code=2): Print an error message and exit with the given code.

Exit codes
----------
EXIT_USAGE                = 2
EXIT_DATA_MISSING         = 3
EXIT_INVALID_ANNOTATION   = 4
EXIT_IO_ERROR             = 5
EXIT_INVALID_INSTRUCTION  = 7

Notes
-----
- All messages printed include a "(HFinder)" prefix for easy log filtering.
"""

import sys

# ----------------------------------------------------------------------
# Verbosity level:
#   1: Error messages only.
#   2: Errors + warnings.
#   3: Errors + warnings + info.
# ----------------------------------------------------------------------
VERBOSITY = 3

# ----------------------------------------------------------------------
# Standard exit codes for predictable program termination.
# ----------------------------------------------------------------------
EXIT_USAGE = 2
EXIT_DATA_MISSING = 3
EXIT_INVALID_ANNOTATION = 4
EXIT_IO_ERROR = 5
EXIT_INVALID_INSTRUCTION = 7


def set_verbosity(n):
    """
    Set the verbosity level for log output.

    :param n: Verbosity mode (1=errors only, 2=errors+warnings, 3=errors+warnings+info).
    :type n: int
    """
    global VERBOSITY
    if n == 1 or n == 2 or n == 3:
        VERBOSITY = n
    else:
        warn(f"Skipping invalid verbosity mode {n}")


def info(msg):
    """
    Print an informational message if verbosity >= 3.

    :param msg: Message text (without trailing punctuation).
    :type msg: str
    """
    global VERBOSITY
    if VERBOSITY > 2:
        print(f"(HFinder) Info: {msg}.")


def warn(msg):
    """
    Print a warning message if verbosity >= 2.

    :param msg: Message text (without trailing punctuation).
    :type msg: str
    """
    global VERBOSITY
    if VERBOSITY > 1:
        print(f"(HFinder) Warning: {msg}.")


def fail(msg, exit_code=2):
    """
    Print a failure message and terminate the program.

    :param msg: Message text (without trailing punctuation).
    :type msg: str
    :param exit_code: Process exit code (default EXIT_USAGE).
    :type exit_code: int
    """
    print(f"(HFinder) Failure: {msg}.")
    sys.exit(exit_code)

