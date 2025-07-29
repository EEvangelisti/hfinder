import sys

VERBOSITY = 1       # Error messages only.
VERBOSITY = 2       # Errors and warnings.
VERBOSITY = 3       # Errors, warnings and messages.

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
