import sys

VERBOSITY = 1       # Error messages only.
VERBOSITY = 2       # Errors and warnings.
VERBOSITY = 3       # Errors, warnings and messages.


EXIT_USAGE = 2
EXIT_DATA_MISSING = 3
EXIT_INVALID_ANNOTATION = 4
EXIT_IO_ERROR = 5
EXIT_INVALID_ANNOTATION = 6
EXIT_INVALID_INSTRUCTION = 7


def set_verbosity(n):
    global VERBOSITY
    if n == 1 or n == 2 or n == 3:
        VERBOSITY = n
    else:
        warn(f"Skipping invalid verbosity mode {n}")

def info(msg):
    global VERBOSITY
    if VERBOSITY > 2:
        print(f"(HFinder) Info: {msg}.")

def warn(msg):
    global VERBOSITY
    if VERBOSITY > 1:
        print(f"(HFinder) Warning: {msg}.")
    
def fail(msg, exit_code=2):
    print(f"(HFinder) Failure: {msg}.")
    sys.exit(exit_code)
    
def assert_failure(cmd):
    return f"(HFinder) Assert Failure: {cmd}"
