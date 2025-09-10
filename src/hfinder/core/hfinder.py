import argparse
from hfinder.core import hf_log as HF_log
from hfinder.core import hf_train as HF_train
from hfinder.core import hf_dataset as HF_dataset
from hfinder.core import hf_folders as HF_folders
from hfinder.core import hf_predict as HF_predict
from hfinder.session import settings as HF_settings


def preprocess(args=None):
    if args is not None:
        HF_settings.load(args)
        HF_settings.set("running_mode", "preprocess")
    HF_settings.print_summary()
    HF_folders.create_session_folders()
    HF_dataset.generate_training_dataset()


def train(args):
    HF_settings.load(args)
    HF_settings.set("running_mode", "train")
    preprocess()
    HF_train.run()


def predict(args):
    HF_settings.load(args)
    HF_settings.set("running_mode", "predict")
    HF_settings.print_summary()
    HF_folders.create_session_folders()
    HF_predict.run()


def main():
    parser = argparse.ArgumentParser(prog="hfinder", description="Computer vision for plant-microbe interfaces")
    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    # ---- Subcommand: preprocessing ----
    parser_check = subparsers.add_parser("preprocess", help="Generate and validate binary masks")
    HF_settings.define_arguments(parser_check, "preprocess")
    parser_check.set_defaults(func=preprocess)

    # ---- Subcommand: training ----
    parser_train = subparsers.add_parser("train", help="Train YOLOv8 model")
    HF_settings.define_arguments(parser_train, "train")
    parser_train.set_defaults(func=train)

    # ---- Subcommand: training ----
    parser_predict = subparsers.add_parser("predict", help="Predict using YOLOv8")
    HF_settings.define_arguments(parser_predict, "predict")
    parser_predict.set_defaults(func=predict)

    # ---- Parse args and dispatch ----
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':

    main()
