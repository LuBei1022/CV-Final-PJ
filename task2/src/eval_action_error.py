"""Zero-shot 评测：在未见过的环境 D 上算 Action L1 / MSE（离线，不需仿真器）。

对两个模型（act_B 与 act_ABC）各跑一次，对比在 D 上的动作误差，体现跨环境泛化。
另外输出"按 chunk 内时间步分解的 L1"，用于报告里分析 ACT 动作分块在视觉分布
偏移下的鲁棒性。

前提：需要环境 D 的数据清单（split_D.json）。当前老师给的 ABC_D 数据只含训练用的
A/B/C，不含 D —— D 测试数据的来源需先确认（见 README）。一旦有了 D 的分片+清单，
本脚本即可直接用。

用法：
    python src/eval_action_error.py \
        --checkpoint ./outputs/act_B/checkpoints/last \
        --shards-root ./data/calvin_D \
        --split-manifest ./data/env_splits/split_D.json \
        --out ./outputs/eval/act_B_on_D.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils import enable_mps_fallback, get_device, move_batch_to_device

enable_mps_fallback()

from lerobot.configs.types import FeatureType
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.utils import dataset_to_policy_features

import calvin_data as cd


def parse_args():
    p = argparse.ArgumentParser(description="Offline action-error eval on env D.")
    p.add_argument("--checkpoint", required=True, help="训练保存的 checkpoint 目录")
    p.add_argument("--shards-root", required=True, help="环境 D 数据所在分片目录")
    p.add_argument("--split-manifest", required=True, help="环境 D 的 episode 清单")
    p.add_argument("--out", default="./outputs/eval/result.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None, help="只评测前 N 个 batch(调试)")
    p.add_argument("--lang-cond", action="store_true", default=True)
    p.add_argument("--no-lang-cond", dest="lang_cond", action="store_false")
    p.add_argument("--lang-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--lang-dim", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    return p.parse_args()


def build_delta_timestamps(cfg, fps):
    def md(indices):
        return [0] if indices is None else [i / fps for i in indices]
    dt = {"action": md(cfg.action_delta_indices)}
    dt |= {k: md(cfg.observation_delta_indices) for k in cfg.image_features}
    return dt


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = cd.load_manifest(args.split_manifest)
    meta = cd.first_shard_metadata(args.shards_root, manifest)

    # 特征 / 语言条件（必须与训练时一致）
    features = cd.filter_policy_features(dataset_to_policy_features(meta.features))
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    lang_encoder, emb_dim, stats = None, 0, meta.stats
    if args.lang_cond:
        lang_encoder = cd.LangEncoder(args.lang_model, device=device.type, dim=args.lang_dim)
        emb_dim = lang_encoder.out_dim
        input_features = cd.augment_state_feature(input_features, emb_dim)
        stats = cd.augment_state_stats(meta.stats, emb_dim)

    # 加载策略权重；前后处理用相同 stats 重建（避免依赖保存格式的加载 API）
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    if hasattr(cfg, "device"):
        cfg.device = device.type
    policy = ACTPolicy.from_pretrained(args.checkpoint)  # VERIFY: 加载 API
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.eval()
    policy.to(device)

    delta_timestamps = build_delta_timestamps(cfg, meta.fps)
    dataset = cd.EnvFilteredDataset(args.shards_root, manifest, delta_timestamps, lang_encoder)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    # 累计两类指标：
    #   1) 整体 Action L1 / MSE（主指标）
    #   2) 按 chunk 内时间步的 L1（用于 action chunking 鲁棒性分析）
    sum_l1 = sum_mse = n_frames = 0.0
    per_pos_l1, per_pos_cnt = None, None

    for bi, batch in enumerate(loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        batch = preprocessor(batch)
        batch = move_batch_to_device(batch, device)

        gt = batch["action"]  # (B, chunk, action_dim)（按 delta_timestamps 加载的真值块）

        # 预测动作块。不同 LeRobot 版本接口名可能不同，按优先级尝试。 # VERIFY
        pred = None
        for name in ("predict_action_chunk", "generate_actions"):
            fn = getattr(policy, name, None)
            if callable(fn):
                pred = fn(batch)
                break
        if pred is None:
            # 退路：用训练前向返回的 l1_loss 作为整体动作误差
            loss, info = policy.forward(batch)
            l1 = float(info.get("l1_loss", loss)) if isinstance(info, dict) else float(loss)
            bs = gt.shape[0]
            sum_l1 += l1 * bs
            n_frames += bs
            continue

        pred = pred[..., : gt.shape[-1]] if pred.shape[-1] != gt.shape[-1] else pred
        diff = (pred - gt).abs()                      # (B, chunk, dim)
        sum_l1 += diff.mean(dim=(1, 2)).sum().item()
        sum_mse += ((pred - gt) ** 2).mean(dim=(1, 2)).sum().item()
        n_frames += gt.shape[0]

        pos_l1 = diff.mean(dim=(0, 2))                # (chunk,) 每个时间步的平均 L1
        if per_pos_l1 is None:
            per_pos_l1 = torch.zeros_like(pos_l1)
            per_pos_cnt = 0
        per_pos_l1 += pos_l1 * gt.shape[0]
        per_pos_cnt += gt.shape[0]

    result = {
        "checkpoint": args.checkpoint,
        "eval_manifest": args.split_manifest,
        "n_frames": int(n_frames),
        "action_l1": sum_l1 / max(n_frames, 1),
        "action_mse": (sum_mse / max(n_frames, 1)) if sum_mse else None,
    }
    if per_pos_l1 is not None and per_pos_cnt:
        result["per_chunk_position_l1"] = (per_pos_l1 / per_pos_cnt).cpu().tolist()

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] 写出 {out}")


if __name__ == "__main__":
    main()
