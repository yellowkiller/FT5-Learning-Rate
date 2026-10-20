import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    predicted = outputs.argmax(dim=1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return correct, total

