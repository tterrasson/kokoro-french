"""Training progress helpers."""

from __future__ import annotations

from numbers import Number

import torch
from tqdm.auto import tqdm


def make_progress_bar(iterable, *, total, desc, disable=False):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="it",
        dynamic_ncols=True,
        leave=True,
        disable=disable,
    )


def metric_value(value):
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().item()
    return float(value)


def metric_postfix(**metrics):
    return {key: f"{metric_value(value):.5f}" for key, value in metrics.items()}
