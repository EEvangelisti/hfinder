import argparse
import hfinder_log as HFinder_log
import hfinder_train as HFinder_train
import hfinder_hyphae as HFinder_hyphae
import hfinder_folders as HFinder_folders
import hfinder_settings as HFinder_settings
import hfinder_preprocess as HFinder_preprocess


def generate_dataset(args, silent=False):
    HFinder_settings.load(args)
    print("(HFinder) Creating folders...")
    folder_tree = HFinder_folders.create_training_folders()
    print("(HFinder) Generating dataset...")
    HFinder_hyphae.generate_dataset(folder_tree)
    if not silent:
        print("(HFinder) OK")
    return folder_tree


def train(args):
    HFinder_settings.load(args)
    HFinder_log.info("Creating folders...")
    folder_tree = HFinder_folders.create_training_folders()
    HFinder_preprocess.generate_training_dataset(folder_tree)
    HFinder_train.train_yolo_model(folder_tree)


def main():
    parser = argparse.ArgumentParser(prog="hfinder", description="HFinder CLI for hyphae datasets")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    # ---- Subcommand: check_masks ----
    parser_check = subparsers.add_parser("check", help="Generate and validate binary masks")
    parser_check.set_defaults(func=generate_dataset)

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
