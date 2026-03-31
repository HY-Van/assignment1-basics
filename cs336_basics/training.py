from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import BinaryIO, IO

import numpy as np
import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    log_norm = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_norm - target_logits).mean()


def get_batch(
    dataset: np.ndarray | Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    data = torch.as_tensor(dataset, dtype=torch.long)
    max_start = data.shape[0] - context_length
    if max_start <= 0:
        raise ValueError("Dataset must be longer than context_length.")

    starts = torch.randint(0, max_start, (batch_size,))
    offsets = torch.arange(context_length)
    x = data[starts.unsqueeze(1) + offsets]
    y = data[starts.unsqueeze(1) + offsets + 1]
    return x.to(device), y.to(device)


def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(torch.sum(grad.detach() * grad.detach()) for grad in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return

    for grad in grads:
        grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate

    decay_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])
