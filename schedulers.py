import torch


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, *, epochs: int, steps_per_epoch: int):
    name = name.lower()
    if name in {"constant", "fixed", "none"}:
        return None, "epoch"
    if name in {"step", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1), "epoch"
    if name in {"multistep", "multisteplr"}:
        milestones = [max(epochs // 2, 1), max(epochs * 3 // 4, 1)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1), "epoch"
    if name in {"cosine", "cosineannealing"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), "epoch"
    if name in {"warmrestart", "cosine_restart", "cosinewarmrestarts"}:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(epochs // 4, 1), T_mult=2
        ), "epoch"
    if name in {"cyclic", "cycliclr"}:
        base_lr = optimizer.param_groups[0]["lr"] / 10
        max_lr = optimizer.param_groups[0]["lr"]
        step_size_up = max(steps_per_epoch * 2, 1)
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode="triangular2",
            cycle_momentum=True,
        ), "batch"
    if name in {"onecycle", "onecyclelr"}:
        max_lr = optimizer.param_groups[0]["lr"]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch
        ), "batch"
    raise ValueError(f"Unknown scheduler: {name}")

