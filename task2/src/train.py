from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")            # 本地数据,禁止连 hub(否则 401)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_home")  # 缓存放数据盘,避免系统盘写满(Errno 28)

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


# ------------------------------ 工具 ------------------------------
def get_device(pref=None):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=(device.type == "cuda")) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# 这份数据用裸字段名,需映射成 lerobot ACT 期望的标准键名
KEY_MAP = {
    "actions": "action",                              # 动作输出
    "image": "observation.images.image",              # 主相机
    "wrist_image": "observation.images.wrist_image",  # 手腕相机
    "state": "observation.state",                     # 本体状态(ACT 必需)
}
REV_MAP = {v: k for k, v in KEY_MAP.items()}          # 标准键 -> 数据集键


def _map_key(k, mapping):
    # 同时处理 padding 掩码键,如 actions_is_pad -> action_is_pad
    if k.endswith("_is_pad"):
        base = k[: -len("_is_pad")]
        return mapping.get(base, base) + "_is_pad"
    return mapping.get(k, k)


def remap_keys(d, mapping):
    return {_map_key(k, mapping): v for k, v in d.items()}


class RemapDataset(Dataset):
    """把底层数据集每个样本的键名改成 lerobot 标准名。"""
    def __init__(self, base, key_map):
        self.base, self.key_map = base, key_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return remap_keys(self.base[i], self.key_map)


def make_delta(indices, fps):
    return [0] if indices is None else [i / fps for i in indices]


def split_episodes(n, val_frac, seed):
    eps = list(range(n))
    random.Random(seed).shuffle(eps)
    k = max(1, int(n * val_frac)) if val_frac > 0 else 0
    return eps[k:], eps[:k]


def combined_stats(metas):
    """多个 split 的归一化统计合并;失败则退回用第一个。"""
    if len(metas) == 1:
        return metas[0].stats
    try:
        from lerobot.datasets.compute_stats import aggregate_stats
        return aggregate_stats([m.stats for m in metas])
    except Exception as e:
        print(f"[stats] 合并失败({e}),退回用第一个 split 的统计")
        return metas[0].stats


class Tracker:
    def __init__(self, kind, project, run_name, config):
        self.kind, self.run = kind, None
        if kind == "swanlab":
            import swanlab
            self.run = swanlab.init(project=project, experiment_name=run_name, config=config)

    def log(self, data, step):
        if self.run is not None:
            self.run.log(data, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()


@torch.no_grad()
def evaluate(policy, preprocessor, loader, device, max_batches):
    # 注:ACT 在 eval 模式下不跑 VAE 编码器(mu/logvar=None),会让 KL 计算报错;
    # 这里保持 train 模式(配合 no_grad)来算验证 Action L1。
    policy.train()
    tot, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = to_device(preprocessor(batch), device)
        loss, info = policy.forward(batch)
        l1 = float(info["l1_loss"]) if isinstance(info, dict) and "l1_loss" in info else float(loss)
        bs = len(next(v for v in batch.values() if torch.is_tensor(v)))
        tot += l1 * bs
        n += bs
    policy.train()
    return tot / max(n, 1)


# ------------------------------ 主程序 ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="一个或多个 split 目录(绝对路径)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="每个 split 只用前 N 个 episode(冒烟测试用,大幅加快首次加载)")
    p.add_argument("--eval-freq", type=int, default=2000)
    p.add_argument("--eval-max-batches", type=int, default=50)
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=10000)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--tracker", default="swanlab", choices=["swanlab", "none"])
    p.add_argument("--project", default="calvin-act")
    p.add_argument("--run-name", default="act")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    data_dirs = [Path(d) for d in args.data]
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[device] {device} | data={[d.name for d in data_dirs]}")

    # 1) 元信息 + 特征(以第一个 split 为代表;多个 split 特征一致)
    metas = [LeRobotDatasetMetadata(d.name, root=d) for d in data_dirs]
    fps = metas[0].fps
    raw_features = dataset_to_policy_features(metas[0].features)
    # 只保留需要的字段并改成标准键名(image/wrist_image/actions -> 标准名)
    features = {KEY_MAP[k]: ft for k, ft in raw_features.items() if k in KEY_MAP}
    # lerobot 不会把裸键 "state" 当作策略输入,这里手动补成 observation.state(ACT 必需)
    if "state" in metas[0].features and "observation.state" not in features:
        state_shape = tuple(metas[0].features["state"]["shape"])
        features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=state_shape)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    print(f"[features] inputs={list(input_features)} outputs={list(output_features)}")

    # 2) ACT 配置 / 策略 / 前后处理
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    if hasattr(cfg, "device"):
        cfg.device = device.type
    policy = ACTPolicy(cfg)
    stats = remap_keys(combined_stats(metas), KEY_MAP)   # 统计也改成标准键名
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.train()
    policy.to(device)

    # 3) delta_timestamps —— LeRobotDataset 按"数据集原始键名"加载,故这里用 REV_MAP 映回
    action_std = next(iter(output_features))             # "action"
    # 只给动作加时间维(动作分块需要);ACT 用单帧观测,图像/状态不加时间维,
    # 否则图像会变成 5D (B,1,C,H,W) 让 conv2d 报错。
    delta = {REV_MAP[action_std]: make_delta(cfg.action_delta_indices, fps)}   # "actions"

    # 4) 数据集(单个或多个 split;各自留出验证 episode 再拼接)
    train_parts, val_parts = [], []
    for d, meta in zip(data_dirs, metas):
        n_ep = meta.total_episodes if args.max_episodes is None \
            else min(args.max_episodes, meta.total_episodes)
        tr, va = split_episodes(n_ep, args.val_frac, args.seed)
        train_parts.append(LeRobotDataset(d.name, root=d, episodes=tr, delta_timestamps=delta))
        if va:
            val_parts.append(LeRobotDataset(d.name, root=d, episodes=va, delta_timestamps=delta))
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    train_ds = RemapDataset(train_ds, KEY_MAP)           # 统一改成标准键名
    val_loader = None
    if val_parts:
        val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
        val_ds = RemapDataset(val_ds, KEY_MAP)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
    print(f"[data] train_frames={len(train_ds)}" + (f" val_frames={len(val_ds)}" if val_loader else ""))

    # 5) 优化器 / dataloader
    preset = cfg.get_optimizer_preset()
    if args.lr is not None and hasattr(preset, "lr"):
        preset.lr = args.lr
    optimizer = preset.build(policy.parameters())
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)

    # 6) 记录配置(报告超参数表用)
    cfg_dump = {"data": [str(d) for d in data_dirs], "steps": args.steps,
                "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
                "device": device.type, "chunk_size": getattr(cfg, "chunk_size", None)}
    (out / "train_config.json").write_text(json.dumps(cfg_dump, indent=2, default=str))
    tracker = Tracker(args.tracker, args.project, args.run_name, cfg_dump)

    def save(name):
        ck = out / "checkpoints" / name
        policy.save_pretrained(ck)
        preprocessor.save_pretrained(ck)
        postprocessor.save_pretrained(ck)
        return ck

    # 7) 训练循环
    print(f"[train] steps={args.steps} batch={args.batch_size}")
    best, step, done, t0 = float("inf"), 0, False, time.time()
    while not done:
        for batch in loader:
            batch = to_device(preprocessor(batch), device)
            loss, info = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                log = {"loss": float(loss.item())}
                if isinstance(info, dict):
                    for k, v in info.items():
                        try:
                            log[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                log["sps"] = (step + 1) / (time.time() - t0)
                tracker.log(log, step)
                print(f"step {step:>7d} | " + " | ".join(f"{k}={v:.4f}" for k, v in log.items()))

            if val_loader is not None and step > 0 and step % args.eval_freq == 0:
                v = evaluate(policy, preprocessor, val_loader, device, args.eval_max_batches)
                tracker.log({"val_action_l1": v}, step)
                print(f"step {step:>7d} | val_action_l1={v:.4f} (best={best:.4f})")
                if v < best:
                    best = v
                    save("best")
                    (out / "best_metric.json").write_text(json.dumps({"step": step, "val_action_l1": v}))
                    print(f"[best] -> checkpoints/best (val_action_l1={v:.4f})")

            if step > 0 and step % args.save_freq == 0:
                print(f"[ckpt] {save(f'{step:07d}')}")

            step += 1
            if step >= args.steps:
                done = True
                break

    save("last")
    if val_loader is None:
        save("best")
    print(f"[done] 提交用最优权重: {out/'checkpoints'/'best'}")
    tracker.finish()


if __name__ == "__main__":
    main()
