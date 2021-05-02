"""Uses a pretrained model to generate name proposals for the training, validation and test sets"""

import argparse
from glob import glob
import os
import torch
from transformers import AutoTokenizer


from data import load_data
from model import SpainAICollator, generate_names, load_model
from postprocessing import save_submission


def main(model_path, model_class, model_name, do_train=False, do_val=False, do_test=False, ensemble=False, batchsize=2):
    # If not a ensemble, just generate names for the single model provided
    if not ensemble:
        generate_names_model(model_path, model_class, model_name, do_train, do_val, do_test)
        return

    # If a ensemble, gather all models
    ensemble_paths = ["/".join(os.path.split(x)[:-1]) for x in glob(f"{model_path}/*/pytorch_model.bin")]
    ensemble_numbers = [int(os.path.split(path)[-1]) for path in ensemble_paths]

    # Generate names for all elements in the ensemble
    for path, number in zip(ensemble_paths, ensemble_numbers):
        print(f"Generating name proposals for ensemble model {path} (number {number})")
        generate_names_model(path, model_class, f"{model_name}_{number}", do_train, do_val, do_test, batchsize, data_fold=number)
        torch.cuda.empty_cache()


def generate_names_model(model_path, model_class, model_name, do_train=False, do_val=False, do_test=False, batchsize=2, data_fold=0):
    """Generate predictions for a given model"""
    # Load model
    model = load_model(model_path, model_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_class)

    # Prepare collator
    collator = SpainAICollator(tokenizer, model)

    # Load data
    train, val, test = load_data(random_state=data_fold)

    # Prepare sets to work
    datasets = []
    if do_train: datasets.append(("train", train))
    if do_val: datasets.append(("val", val))
    if do_test: datasets.append(("test", test))

    # Generate names
    for dataname, data in datasets:
        print(f"Generating name proposals for dataset {dataname}")
        names = generate_names(model, tokenizer, collator, data["description"], batchsize=batchsize, num_sequences=30, max_candidates=10)
        save_submission(names, f"{model_name}_{dataname}", zip=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Uses a pretrained model to generate name proposals for the training, validation and test sets')
    parser.add_argument('model_path', type=str, help='path to pretrained model, or to model folder in the case of an ensemble')
    parser.add_argument('model_class', type=str, help='class of model (for tokenizer)')
    parser.add_argument('model_name', type=str, help='suffix to use to save output files, including path')
    parser.add_argument('--train', action='store_true', help='generate name proposals for training data')
    parser.add_argument('--val', action='store_true', help='generate name proposals for validation data')
    parser.add_argument('--test', action='store_true', help='generate name proposals for test data')
    parser.add_argument('--ensemble', action='store_true', help='whether the model at the path is an ensemble (generate predictions for all elements)')
    parser.add_argument('--batchsize', type=int, default=2, help='batch size for generation')
    args = parser.parse_args()
    main(args.model_path, args.model_class, args.model_name, args.train, args.val, args.test, args.ensemble, args.batchsize)
