import sys

def main() -> int:
    from hfinder.main.main import main as _main
    return _main()

def annot2images() -> int:
    from hfinder.extra.annot2images import main as _main
    return _main()

def annot2signal() -> int:
    from hfinder.extra.annot2signal import main as _main
    return _main()

def annot2distances() -> int:
    from hfinder.extra.annot2distances import main as _main
    return _main()

if __name__ == "__main__":
    sys.exit(main())
