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

# ---- 与 train.py 一致的字段重映射 ----
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="训练保存的 checkpoints/best 目录")
    p.add_argument("--train-data", nargs="+", required=True,
                   help="该模型训练用的 split(用于复原归一化统计;B 模型填 splitB,联合模型填 A B C)")
    p.add_argument("--eval-data", required=True, help="splitD 目录")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-episodes", type=int, default=None, help="只评测前 N 个 D episode(加速)")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")

    # 1) 加载训练好的策略(含 config + 权重)
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.train()  # ACT 用 train 模式前向才能算 l1_loss(eval 模式 VAE 关闭会报错)
    cfg = policy.config

    # 2) 用"训练数据"的统计复原归一化(必须与训练一致)
    train_metas = [LeRobotDatasetMetadata(Path(d).name, root=d) for d in args.train_data]
    stats = remap_keys(combined_stats(train_metas), KEY_MAP)
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=stats)

    # 3) 评测数据集(环境 D),只给动作加时间维
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

    # 4) 遍历算 Action L1
    sum_l1, n = 0.0, 0
    for batch in loader:
        batch = to_device(preprocessor(batch), device)
        loss, info = policy.forward(batch)
        l1 = float(info["l1_loss"]) if isinstance(info, dict) and "l1_loss" in info else float(loss)
        bs = int(batch["action"].shape[0])
        sum_l1 += l1 * bs
        n += bs

    result = {
        "checkpoint": args.checkpoint,
        "train_data": [str(d) for d in args.train_data],
        "eval_data": str(d_dir),
        "n_frames": n,
        "action_l1": sum_l1 / max(n, 1),
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] 写出 {out}")


if __name__ == "__main__":
    main()
