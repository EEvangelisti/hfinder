import sys

def main() -> int:
    from hfinder.core.hfinder import main as _main
    return _main()

def annot2images() -> int:
    from hfinder.toolbox.annot2images import main as _main
    return _main()

if __name__ == "__main__":
    sys.exit(main())
