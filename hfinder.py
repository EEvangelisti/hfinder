import argparse
import hfinder_log as HFinder_log
import hfinder_train as HFinder_train
import hfinder_dataset as HFinder_dataset
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings


def preprocess(args):
    HFinder_settings.load(args)
    HFinder_settings.print_summary()
    HFinder_folders.create_session_folders()
    HFinder_dataset.generate_training_dataset()


def train(args):
    preprocess(args)
    HFinder_train.run()


def main():
    parser = argparse.ArgumentParser(prog="hfinder", description="Computer vision for plant-microbe interfaces")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    # ---- Subcommand: preprocessing ----
    parser_check = subparsers.add_parser("preprocess", help="Generate and validate binary masks")
    HFinder_settings.define_arguments(parser_check, "preprocess")
    parser_check.set_defaults(func=preprocess)

    # ---- Subcommand: training ----
    parser_train = subparsers.add_parser("train", help="Train YOLOv8 model on hyphae dataset")
    HFinder_settings.define_arguments(parser_train, "train")
    parser_train.set_defaults(func=train)

    # ---- Parse args and dispatch ----
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':

    main()
