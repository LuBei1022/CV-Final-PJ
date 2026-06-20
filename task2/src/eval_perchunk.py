#!/usr/bin/env python
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_home")

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

KEY_MAP = {
    "actions": "action",
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
}
REV_MAP = {v: k for k, v in KEY_MAP.items()}


def _map_key(k, mapping):
    if k.endswith("_is_pad"):
        return mapping.get(k[: -len("_is_pad")], k[: -len("_is_pad")]) + "_is_pad"
    return mapping.get(k, k)


def remap_keys(d, mapping):
    return {_map_key(k, mapping): v for k, v in d.items()}


class RemapDataset(Dataset):
    def __init__(self, base, key_map):
        self.base, self.key_map = base, key_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return remap_keys(self.base[i], self.key_map)


def get_device(pref=None):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_delta(indices, fps):
    return [0] if indices is None else [i / fps for i in indices]


def combined_stats(metas):
    if len(metas) == 1:
        return metas[0].stats
    try:
        from lerobot.datasets.compute_stats import aggregate_stats
        return aggregate_stats([m.stats for m in metas])
    except Exception:
        return metas[0].stats


def predict_chunk(policy, batch):
    """尽量兼容不同 lerobot 版本的"预测动作块"接口,返回 (B, T, action_dim)。"""
    for name in ("predict_action_chunk", "generate_actions", "predict_action"):
        fn = getattr(policy, name, None)
        if callable(fn):
            out = fn(batch)
            return out
    raise AttributeError("找不到预测动作块的方法(predict_action_chunk 等),请检查 lerobot 版本。")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-data", nargs="+", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()                      # 预测用 eval 模式(从先验采样,不跑 VAE 编码器)
    cfg = policy.config

    train_metas = [LeRobotDatasetMetadata(Path(d).name, root=d) for d in args.train_data]
    stats = remap_keys(combined_stats(train_metas), KEY_MAP)
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=stats)

    d_dir = Path(args.eval_data)
    d_meta = LeRobotDatasetMetadata(d_dir.name, root=d_dir)
    action_std = next(iter(cfg.output_features))
    delta = {REV_MAP[action_std]: make_delta(cfg.action_delta_indices, d_meta.fps)}

    n_ep = d_meta.total_episodes if args.max_episodes is None \
        else min(args.max_episodes, d_meta.total_episodes)
    base = LeRobotDataset(d_dir.name, root=d_dir, episodes=list(range(n_ep)), delta_timestamps=delta)
    ds = RemapDataset(base, KEY_MAP)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[eval] env D frames={len(ds)}")

    pos_sum = None   # 每个 chunk 位置的累计 L1
    count = 0
    for batch in loader:
        gt = batch["action"].clone().to(device)        # 原始动作块 (B, T, dim)
        pbatch = to_device(preprocessor(batch), device)
        pred = predict_chunk(policy, pbatch)            # (B, T, dim)
        if not torch.is_tensor(pred):
            pred = pred[0] if isinstance(pred, (tuple, list)) else torch.as_tensor(pred)
        pred = pred.to(device).float()
        T = min(pred.shape[1], gt.shape[1])
        diff = (pred[:, :T] - gt[:, :T]).abs().mean(dim=2)   # (B, T) 对动作维取平均
        s = diff.sum(dim=0)                                  # (T,)
        pos_sum = s if pos_sum is None else pos_sum + s
        count += gt.shape[0]

    per_pos = (pos_sum / max(count, 1)).cpu().tolist()
    result = {
        "checkpoint": args.checkpoint,
        "eval_data": str(d_dir),
        "n_episodes": int(n_ep),
        "action_l1": float(sum(per_pos) / len(per_pos)),     # chunk 整体平均(原始动作单位)
        "per_chunk_position_l1": per_pos,                    # 逐时间步 L1,供画曲线
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] 整体 chunk L1={result['action_l1']:.4f}, 写出 {out}")


if __name__ == "__main__":
    main()
