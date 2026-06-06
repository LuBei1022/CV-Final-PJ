"""训练 ACT 策略（基础模型 = 仅环境 B；联合模型 = A+B+C）。

同一个脚本训练两个模型，唯一区别是 --split-manifest：
  - 基础模型：--split-manifest ./data/env_splits/split_B.json   --output-dir ./outputs/act_B
  - 联合模型：--split-manifest ./data/env_splits/split_ABC.json --output-dir ./outputs/act_ABC
其余超参数逐字相同，保证两个模型严格可比（题目要求）。

设计要点
- 设备不写死：cuda > mps > cpu（utils.get_device）。
- 全相对路径。
- 数据：跨 4 分片按环境清单过滤 + 语言条件（见 calvin_data）。
- 用 LeRobot 内置 ACT（满足"用框架自带 ACT"）。

本机小步验证（Mac, 自动 mps）：
    python src/train_base.py \
        --shards-root ./data/calvin_raw \
        --split-manifest ./data/env_splits/split_B.json \
        --output-dir ./outputs/act_B \
        --steps 200 --batch-size 4 --num-workers 0 --tracker none

GPU 服务器正式训练（自动 cuda）：
    python src/train_base.py \
        --shards-root ./data/calvin_raw \
        --split-manifest ./data/env_splits/split_B.json \
        --output-dir ./outputs/act_B \
        --steps 100000 --batch-size 32 --tracker swanlab \
        --project calvin-act --run-name act_B
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils import enable_mps_fallback, get_device, move_batch_to_device, set_seed

enable_mps_fallback()

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

import calvin_data as cd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ACT on CALVIN (env-filtered, language-conditioned).")

    # 数据（相对路径）
    p.add_argument("--shards-root", default="./data/calvin_raw",
                   help="下载下来的 4 个分片所在目录")
    p.add_argument("--split-manifest", required=True,
                   help="环境清单：split_B.json（基础）或 split_ABC.json（联合）")

    # 输出
    p.add_argument("--output-dir", required=True, help="checkpoint/日志输出目录")

    # 语言条件
    p.add_argument("--lang-cond", action="store_true", default=True,
                   help="启用语言条件（默认开）")
    p.add_argument("--no-lang-cond", dest="lang_cond", action="store_false",
                   help="关闭语言条件（退化为纯视觉+状态，调试用）")
    p.add_argument("--lang-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--lang-dim", type=int, default=None,
                   help="语言向量截断维度；不填用模型原始维(MiniLM=384)")

    # 训练超参数（两个模型必须一致；记入报告表格）
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="覆盖 ACT 优化器预设学习率")
    p.add_argument("--chunk-size", type=int, default=None, help="ACT 动作分块长度")
    p.add_argument("--num-workers", type=int, default=4, help="Mac 上建议 0")
    p.add_argument("--seed", type=int, default=1000)

    # 日志/保存
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=10_000, help="周期性保存 checkpoint 的步频")

    # 验证集 & 最优权重
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="从清单里按 episode 留出的验证比例(0 关闭验证)")
    p.add_argument("--eval-freq", type=int, default=2_000, help="每多少步在验证集上评一次")
    p.add_argument("--eval-max-batches", type=int, default=50,
                   help="每次验证最多评多少个 batch(控制耗时)")

    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--tracker", default="swanlab", choices=["swanlab", "none"])
    p.add_argument("--project", default="calvin-act")
    p.add_argument("--run-name", default="act")
    return p.parse_args()


class Tracker:
    """SwanLab 实验记录（或 none 关闭）。"""

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


def build_delta_timestamps(cfg, fps):
    def md(indices):
        return [0] if indices is None else [i / fps for i in indices]
    dt = {"action": md(cfg.action_delta_indices)}
    dt |= {k: md(cfg.observation_delta_indices) for k in cfg.image_features}
    return dt


def split_train_val(manifest, val_frac, seed):
    """按 episode 把清单切成 train / val（验证用整段 episode，避免帧级泄漏）。"""
    if val_frac <= 0:
        return manifest, []
    m = manifest[:]
    random.Random(seed).shuffle(m)
    n_val = max(1, int(len(m) * val_frac))
    return m[n_val:], m[:n_val]


@torch.no_grad()
def evaluate_val(policy, preprocessor, loader, device, max_batches):
    """在验证集上算平均 Action L1（用 forward 返回的 l1_loss）。

    注意：ACT 是 CVAE，其 VAE 编码器只在 train 模式下运行并产出 mu/log_sigma，
    forward 的损失计算依赖它们。故这里保持 train 模式（配合 @torch.no_grad 不更新
    权重），否则 eval 模式下 mu/log_sigma=None 会在算 KL 项时报错。
    """
    policy.train()
    tot, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        batch = preprocessor(batch)
        batch = move_batch_to_device(batch, device)
        loss, info = policy.forward(batch)
        l1 = float(info["l1_loss"]) if isinstance(info, dict) and "l1_loss" in info else float(loss)
        bs = int(batch["action"].shape[0])
        tot += l1 * bs
        n += bs
    policy.train()
    return tot / max(n, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[device] {device} | lang_cond={args.lang_cond}")

    out = Path(args.output_dir).resolve()
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    manifest = cd.load_manifest(args.split_manifest)
    train_manifest, val_manifest = split_train_val(manifest, args.val_frac, args.seed)
    print(f"[data] manifest={args.split_manifest} | episodes={len(manifest)} "
          f"(train={len(train_manifest)}, val={len(val_manifest)})")

    # 1) 代表性元信息（fps / features / stats）
    meta = cd.first_shard_metadata(args.shards_root, train_manifest)
    from lerobot.datasets.utils import dataset_to_policy_features
    features = cd.filter_policy_features(dataset_to_policy_features(meta.features))
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    # 2) 语言编码器 + 增广 state 维度与统计
    lang_encoder, emb_dim, stats = None, 0, meta.stats
    if args.lang_cond:
        lang_encoder = cd.LangEncoder(args.lang_model, device=device.type, dim=args.lang_dim)
        emb_dim = lang_encoder.out_dim
        input_features = cd.augment_state_feature(input_features, emb_dim)
        stats = cd.augment_state_stats(meta.stats, emb_dim)
        print(f"[lang] model={args.lang_model} emb_dim={emb_dim} -> state 维度 +{emb_dim}")

    # 3) ACT 配置 / 策略 / 前后处理
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    if hasattr(cfg, "device"):
        cfg.device = device.type
    if args.chunk_size is not None and hasattr(cfg, "chunk_size"):
        cfg.chunk_size = args.chunk_size
        if hasattr(cfg, "n_action_steps"):
            cfg.n_action_steps = args.chunk_size

    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.train()
    policy.to(device)

    # 4) 数据集（跨分片过滤 + 语言条件）
    delta_timestamps = build_delta_timestamps(cfg, meta.fps)
    train_dataset = cd.EnvFilteredDataset(args.shards_root, train_manifest, delta_timestamps, lang_encoder)
    print(f"[data] 训练样本帧: {len(train_dataset)}")
    val_loader = None
    if val_manifest:
        val_dataset = cd.EnvFilteredDataset(args.shards_root, val_manifest, delta_timestamps, lang_encoder)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False)
        print(f"[data] 验证样本帧: {len(val_dataset)}")

    # 5) 优化器 / dataloader
    preset = cfg.get_optimizer_preset()
    if args.lr is not None and hasattr(preset, "lr"):
        preset.lr = args.lr
    optimizer = preset.build(policy.parameters())
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=True)

    # 6) 记录配置（供报告超参数表 / 复现）
    run_config = {
        "split_manifest": args.split_manifest, "shards_root": args.shards_root,
        "device": device.type, "steps": args.steps, "batch_size": args.batch_size,
        "lr_override": args.lr, "seed": args.seed, "lang_cond": args.lang_cond,
        "lang_model": args.lang_model if args.lang_cond else None, "emb_dim": emb_dim,
        "act_config": dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else str(cfg),
    }
    (out / "train_config.json").write_text(json.dumps(run_config, indent=2, default=str))
    tracker = Tracker(args.tracker, args.project, args.run_name, run_config)

    def save_ckpt(name):
        ck = out / "checkpoints" / name
        policy.save_pretrained(ck)
        preprocessor.save_pretrained(ck)
        postprocessor.save_pretrained(ck)
        return ck

    # 7) 训练循环
    print(f"[train] steps={args.steps} batch={args.batch_size}")
    best_val = float("inf")
    step, done, t0 = 0, False, time.time()
    while not done:
        for batch in loader:
            batch = preprocessor(batch)
            batch = move_batch_to_device(batch, device)
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
                tracker.log(log, step=step)
                print(f"step {step:>7d} | " + " | ".join(f"{k}={v:.4f}" for k, v in log.items()))

            # 验证 + 保存最优权重（老师要求交"最优模型"）
            if val_loader is not None and step > 0 and step % args.eval_freq == 0:
                val_l1 = evaluate_val(policy, preprocessor, val_loader, device, args.eval_max_batches)
                tracker.log({"val_action_l1": val_l1}, step=step)
                print(f"step {step:>7d} | val_action_l1={val_l1:.4f} (best={best_val:.4f})")
                if val_l1 < best_val:
                    best_val = val_l1
                    save_ckpt("best")
                    (out / "best_metric.json").write_text(
                        json.dumps({"step": step, "val_action_l1": val_l1}, indent=2))
                    print(f"[best] 新最优 val_action_l1={val_l1:.4f} -> checkpoints/best")

            if step > 0 and step % args.save_freq == 0:
                print(f"[ckpt] {save_ckpt(f'{step:07d}')}")

            step += 1
            if step >= args.steps:
                done = True
                break

    save_ckpt("last")
    # 没开验证时，把 last 也复制成 best，保证一定有可提交的"最优"权重
    if val_loader is None:
        save_ckpt("best")
        print("[best] 未启用验证，已用最终权重作为 best")
    print(f"[done] checkpoints/last 已保存 | 提交用最优权重: {out/'checkpoints'/'best'}")
    tracker.finish()


if __name__ == "__main__":
    main()
