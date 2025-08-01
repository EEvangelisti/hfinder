import argparse
import hfinder_log as HFinder_log
import hfinder_train as HFinder_train
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings
import hfinder_preprocess as HFinder_preprocess


def generate_training_dataset(args):
    HFinder_settings.load(args)
    folder_tree = HFinder_folders.create_training_folders()
    HFinder_preprocess.generate_training_dataset(folder_tree)
    return folder_tree


def train(args):
    folder_tree = generate_training_dataset(args)
    HFinder_train.train_yolo_model(folder_tree)


def main():
    parser = argparse.ArgumentParser(prog="hfinder", description="HFinder CLI for hyphae datasets")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    # ---- Subcommand: check_masks ----
    parser_check = subparsers.add_parser("check", help="Generate and validate binary masks")
    parser_check.set_defaults(func=generate_training_dataset)

    # ---- Subcommand: train ----
    parser_train = subparsers.add_parser("train", help="Train YOLOv8 model on hyphae dataset")
    parser_train.add_argument("--epochs", type=int,
                              default=HFinder_settings.get("epochs"),
                              help="Number of training epochs")
    parser_train.add_argument("--model", type=str,
                              default=HFinder_settings.get("model"),
                              help="Base model")
    parser_train.set_defaults(func=train)

    # ---- Parse args and dispatch ----
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':

    main()
