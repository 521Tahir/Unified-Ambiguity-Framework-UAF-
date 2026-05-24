"""Minimal DDP utilities shared across all training scripts."""
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ddp_concat_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Gather variable-length tensors across all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return x

    world = dist.get_world_size()
    device = x.device

    n_local = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    n_list = [torch.zeros_like(n_local) for _ in range(world)]
    dist.all_gather(n_list, n_local)
    n_list = [int(t.item()) for t in n_list]
    n_max = max(n_list)

    if x.shape[0] < n_max:
        pad = torch.zeros((n_max - x.shape[0],) + x.shape[1:], device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    gather_list = [torch.zeros_like(x_pad) for _ in range(world)]
    dist.all_gather(gather_list, x_pad)

    return torch.cat([gi[:ni] for gi, ni in zip(gather_list, n_list)], dim=0)
