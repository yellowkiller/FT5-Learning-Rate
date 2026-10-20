"""Evaluate a saved checkpoint without starting training."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from config import ExperimentConfig
from models.resnet import build_model
from train import build_loaders, run_epoch, select_device
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    names = {field.name for field in fields(ExperimentConfig)}
    config = ExperimentConfig(**{k: v for k, v in checkpoint["config"].items() if k in names})
    config.data_dir = args.data_dir
    set_seed(config.seed)
    _, validation, test = build_loaders(config)
    model = build_model(config.model, num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss, acc = run_epoch(model, validation if args.split == "validation" else test,
                          torch.nn.CrossEntropyLoss(), device, quiet=True)
    print(json.dumps({"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
                      "split": args.split, "loss": loss, "accuracy": acc}))


if __name__ == "__main__":
    main()

