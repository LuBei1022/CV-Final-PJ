"""通用工具：设备选择、随机种子、batch 设备搬运。

设备策略：cuda > mps > cpu，绝不写死。
所有调用方都应通过 get_device() 获取设备。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """按优先级返回可用设备：cuda > mps > cpu。

    prefer: 可选，强制指定 "cuda"/"mps"/"cpu"。指定但不可用时回退到自动选择。
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    # 自动选择
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """固定随机种子，保证两次训练（B 与 A+B+C）在相同初始化下可比。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """把 batch 里所有 tensor 搬到目标设备（保留非 tensor 字段）。"""
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=(device.type == "cuda"))
        else:
            out[key] = value
    return out


def enable_mps_fallback() -> None:
    """在 mps 上遇到未实现的算子时回退到 CPU，避免训练中途报错。

    必须在进行任何 torch 运算前设置（本函数在 import torch 之后调用即可，
    因为该环境变量在算子执行时才被读取）。
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
