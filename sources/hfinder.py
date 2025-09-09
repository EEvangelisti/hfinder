import argparse
import hfinder_log as HFinder_log
import hfinder_train as HFinder_train
import hfinder_dataset as HFinder_dataset
import hfinder_folders as HFinder_folders
import hfinder_predict as HFinder_predict
import hfinder_settings as HFinder_settings


def preprocess(args=None):
    if args is not None:
        HFinder_settings.load(args)
        HFinder_settings.set("running_mode", "preprocess")
    HFinder_settings.print_summary()
    HFinder_folders.create_session_folders()
    HFinder_dataset.generate_training_dataset()


def train(args):
    HFinder_settings.load(args)
    HFinder_settings.set("running_mode", "train")
    preprocess()
    HFinder_train.run()


def predict(args):
    HFinder_settings.load(args)
    HFinder_settings.set("running_mode", "predict")
    HFinder_settings.print_summary()
    HFinder_folders.create_session_folders()
    HFinder_predict.run()


def main():
    parser = argparse.ArgumentParser(prog="hfinder", description="Computer vision for plant-microbe interfaces")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    # ---- Subcommand: preprocessing ----
    parser_check = subparsers.add_parser("preprocess", help="Generate and validate binary masks")
    HFinder_settings.define_arguments(parser_check, "preprocess")
    parser_check.set_defaults(func=preprocess)

    # ---- Subcommand: training ----
    parser_train = subparsers.add_parser("train", help="Train YOLOv8 model")
    HFinder_settings.define_arguments(parser_train, "train")
    parser_train.set_defaults(func=train)

    # ---- Subcommand: training ----
    parser_predict = subparsers.add_parser("predict", help="Predict using YOLOv8")
    HFinder_settings.define_arguments(parser_predict, "predict")
    parser_predict.set_defaults(func=predict)

    # ---- Parse args and dispatch ----
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':

    main()
