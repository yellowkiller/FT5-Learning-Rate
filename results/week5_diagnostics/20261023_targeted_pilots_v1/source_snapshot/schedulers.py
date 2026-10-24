import torch


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, *, epochs: int, steps_per_epoch: int, options=None):
    name = name.lower()
    options = {} if options is None else dict(options)
    allowed = {"warmrestart": {"T_0", "T_mult"}, "cyclic": {"base_lr", "step_size_up"},
               "onecycle": {"pct_start"}}
    if options.keys() - allowed.get(name, set()):
        raise ValueError(f"Unsupported scheduler options for {name}: {options}")
    initial_lr = optimizer.param_groups[0]["lr"]
    if name in {"constant", "fixed", "none"}:
        metadata = {
            "scheduler_name": "constant",
            "scheduler_step": "epoch",
            "initial_lr": initial_lr,
            "min_lr": initial_lr,
            "max_lr": initial_lr,
            "scheduler_params": "fixed_lr",
        }
        return None, "epoch", metadata
    if name in {"step", "steplr"}:
        step_size = max(epochs // 3, 1)
        gamma = 0.1
        metadata = {
            "scheduler_name": "step",
            "scheduler_step": "epoch",
            "initial_lr": initial_lr,
            "min_lr": initial_lr * gamma ** ((epochs - 1) // step_size),
            "max_lr": initial_lr,
            "scheduler_params": f"step_size={step_size};gamma={gamma}",
        }
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma), "epoch", metadata
    if name in {"multistep", "multisteplr"}:
        milestones = [max(epochs // 2, 1), max(epochs * 3 // 4, 1)]
        gamma = 0.1
        metadata = {
            "scheduler_name": "multistep",
            "scheduler_step": "epoch",
            "initial_lr": initial_lr,
            "min_lr": initial_lr * (gamma ** len(milestones)),
            "max_lr": initial_lr,
            "scheduler_params": f"milestones={milestones};gamma={gamma}",
        }
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma), "epoch", metadata
    if name in {"cosine", "cosineannealing"}:
        eta_min = 0.0
        metadata = {
            "scheduler_name": "cosine",
            "scheduler_step": "epoch",
            "initial_lr": initial_lr,
            "min_lr": eta_min,
            "max_lr": initial_lr,
            "scheduler_params": f"T_max={epochs};eta_min={eta_min}",
        }
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min), "epoch", metadata
    if name in {"warmrestart", "cosine_restart", "cosinewarmrestarts"}:
        t_0 = options.get("T_0", max(epochs // 4, 1))
        t_mult = options.get("T_mult", 2)
        eta_min = 0.0
        metadata = {
            "scheduler_name": "warmrestart",
            "scheduler_step": "epoch",
            "initial_lr": initial_lr,
            "min_lr": eta_min,
            "max_lr": initial_lr,
            "scheduler_params": f"T_0={t_0};T_mult={t_mult};eta_min={eta_min}",
        }
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
        ), "epoch", metadata
    if name in {"warmup_cosine", "warmupcosine", "linear_warmup_cosine"}:
        warmup_epochs = max(epochs // 20, 1)
        cosine_epochs = max(epochs - warmup_epochs, 1)
        start_factor = 0.01
        eta_min = 0.0
        metadata = {
            "scheduler_name": "warmup_cosine",
            "scheduler_step": "epoch",
            "initial_lr": initial_lr,
            "min_lr": min(initial_lr * start_factor, eta_min),
            "max_lr": initial_lr,
            "scheduler_params": (
                f"warmup_epochs={warmup_epochs};start_factor={start_factor};"
                f"cosine_epochs={cosine_epochs};eta_min={eta_min}"
            ),
        }
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        ), "epoch", metadata
    if name in {"cyclic", "cycliclr"}:
        base_lr = options.get("base_lr", initial_lr / 10)
        max_lr = initial_lr
        step_size_up = options.get("step_size_up", max(steps_per_epoch * 2, 1))
        if not 0 < base_lr < max_lr:
            raise ValueError("CyclicLR requires 0 < base_lr < max_lr")
        mode = "triangular2"
        metadata = {
            "scheduler_name": "cyclic",
            "scheduler_step": "batch",
            "initial_lr": initial_lr,
            "min_lr": base_lr,
            "max_lr": max_lr,
            "scheduler_params": f"base_lr={base_lr};max_lr={max_lr};step_size_up={step_size_up};mode={mode};cycle_momentum=False",
        }
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode,
            cycle_momentum=False,
        ), "batch", metadata
    if name in {"onecycle", "onecyclelr"}:
        max_lr = initial_lr
        div_factor = 25.0
        final_div_factor = 10000.0
        pct_start = options.get("pct_start", 0.3)
        if not 0 < pct_start < 1:
            raise ValueError("OneCycleLR requires 0 < pct_start < 1")
        metadata = {
            "scheduler_name": "onecycle",
            "scheduler_step": "batch",
            "initial_lr": initial_lr,
            "min_lr": max_lr / div_factor / final_div_factor,
            "max_lr": max_lr,
            "scheduler_params": (
                f"max_lr={max_lr};epochs={epochs};steps_per_epoch={steps_per_epoch};"
                f"pct_start={pct_start};div_factor={div_factor};final_div_factor={final_div_factor};cycle_momentum=False"
            ),
        }
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            cycle_momentum=False,
        ), "batch", metadata
    raise ValueError(f"Unknown scheduler: {name}")

