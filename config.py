from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    data_dir: str = "../data"
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"
    model: str = "resnet18"
    optimizer: str = "sgd"
    scheduler: str = "constant"
    batch_size: int = 128
    epochs: int = 200
    momentum: float = 0.9
    weight_decay: float = 5e-4
    initial_lr: float = 0.1
    min_lr: float = 0.0
    max_lr: float = 0.1
    seed: int = 1
    num_workers: int = 0
    validation_size: int = 5000
    device: str = "auto"
    run_name: str = "baseline"
    quiet: bool = False
