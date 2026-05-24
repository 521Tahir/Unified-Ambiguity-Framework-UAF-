"""
Shared training utilities: checkpoint saving, metric logging, early stopping.
Dataset-specific trainers import from here.
"""
import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str, meta: Optional[Dict] = None):
    obj = {"model": model.state_dict()}
    if meta:
        obj["meta"] = meta
    torch.save(obj, path)


def load_checkpoint(model: nn.Module, path: str, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt.get("meta", {})


def save_metrics(metrics: Dict, path: str):
    with open(path, "w") as f:
        json.dump(
            {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()},
            f, indent=2,
        )


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    total_steps: int,
):
    from transformers import get_cosine_schedule_with_warmup

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return opt, sched
