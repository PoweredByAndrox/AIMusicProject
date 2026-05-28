from __future__ import annotations

import os
from typing import Any

import torch


def resolve_torch_device(
    *,
    task_name: str,
    requested: str | None = None,
    allow_cpu: bool = True,
) -> torch.device:
    requested_device = (requested or os.getenv('ML_TORCH_DEVICE') or 'auto').strip().lower()
    require_cuda = os.getenv('REQUIRE_CUDA', '0') == '1'

    if requested_device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if require_cuda and not allow_cpu:
            raise RuntimeError(f"{task_name} requires CUDA but no visible GPU is available")
        return torch.device('cpu')

    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            if require_cuda or not allow_cpu:
                raise RuntimeError(f"{task_name} requested CUDA but no visible GPU is available")
            return torch.device('cpu')
        return torch.device('cuda')

    if requested_device == 'cpu':
        if require_cuda and not allow_cpu:
            raise RuntimeError(f"{task_name} requested CPU while REQUIRE_CUDA=1")
        return torch.device('cpu')

    raise ValueError(f"Unsupported torch device request for {task_name}: {requested_device}")


def visible_cuda_summary() -> dict[str, Any]:
    count = int(torch.cuda.device_count())
    return {
        'cuda_available': bool(torch.cuda.is_available()),
        'device_count': count,
        'device_names': [torch.cuda.get_device_name(index) for index in range(count)],
    }
