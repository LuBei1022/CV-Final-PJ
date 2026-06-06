"""环境 D zero-shot 离线动作误差评测（直读 parquet + padding mask）。"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from utils import enable_mps_fallback, get_device, move_batch_to_device

enable_mps_fallback()

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

import calvin_data as cd

EP_RE = re.compile(r"episode_(\d+)\.parquet$")
FYWANG_STATIC = "observation.images.top"
FYWANG_WRIST = "observation.images.wrist"


def parse_args():
    p = argparse.ArgumentParser(description="Standalone offline action-error eval on env D.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="./outputs/eval/result_D.json")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--lang-cond", action="store_true", default=True)
    p.add_argument("--no-lang-cond", dest="lang_cond", action="store_false")
    p.add_argument("--lang-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--lang-dim", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--tracker", default="none", choices=["swanlab", "none"])
    p.add_argument("--project", default="calvin-act")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def decode_img(cell, size):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def iter_episode_files(data_root):
    return sorted(glob.glob(str(data_root / "data" / "chunk-*" / "episode_*.parquet")),
                  key=lambda s: int(EP_RE.search(s).group(1)))


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    cfg = policy.config
    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
    policy.eval()
    policy.to(device)

    chunk = int(getattr(cfg, "chunk_size", 100))
    img_keys = [k for k in cfg.input_features if k.startswith("observation.images")]
    static_key = next(k for k in img_keys if "wrist" not in k)
    wrist_key = next(k for k in img_keys if "wrist" in k)
    print(f"[cfg] chunk_size={chunk} | static={static_key} wrist={wrist_key} | lang={args.lang_cond}")

    lang_table = None
    if args.lang_cond:
        lang_encoder = cd.LangEncoder(args.lang_model, device=device.type, dim=args.lang_dim)
        lang_table = lang_encoder.table_for(cd.load_task_map(data_root))
        print(f"[lang] emb_dim={lang_encoder.out_dim} tasks={len(lang_table)}")

    files = iter_episode_files(data_root)
    if args.max_episodes:
        files = files[: args.max_episodes]
    print(f"[data] 评测 {len(files)} episodes | stride={args.stride}")

    sum_l1 = sum_mse = denom = 0.0
    n_frames = 0
    per_pos_l1 = per_pos_cnt = None
    buf = {"img": [], "wrist": [], "state": [], "action": [], "valid": []}

    def flush():
        nonlocal sum_l1, sum_mse, denom, n_frames, per_pos_l1, per_pos_cnt
        if not buf["action"]:
            return
        batch = {
            static_key: torch.from_numpy(np.stack(buf["img"])).float(),
            wrist_key: torch.from_numpy(np.stack(buf["wrist"])).float(),
            "observation.state": torch.from_numpy(np.stack(buf["state"])).float(),
            "action": torch.from_numpy(np.stack(buf["action"])).float(),
        }
        valid_lens = buf["valid"][:]
        batch = preprocessor(batch)
        batch = move_batch_to_device(batch, device)
        gt = batch["action"]
        pred = None
        for name in ("predict_action_chunk", "generate_actions"):
            fn = getattr(policy, name, None)
            if callable(fn):
                pred = fn(batch)
                break
        if pred is None:
            policy.train()
            loss, info = policy.forward(batch)
            policy.eval()
            l1 = float(info.get("l1_loss", loss)) if isinstance(info, dict) else float(loss)
            bs = gt.shape[0]
            sum_l1 += l1 * bs
            denom += bs
            n_frames += bs
            for k in buf:
                buf[k].clear()
            return
        m = min(pred.shape[1], gt.shape[1])
        pred, gtc = pred[:, :m], gt[:, :m]
        valid = torch.tensor(valid_lens, device=gt.device).clamp(max=m).float()
        ar = torch.arange(m, device=gt.device).unsqueeze(0)
        mask = (ar < valid.unsqueeze(1)).float()
        pos_l1 = (pred - gtc).abs().mean(dim=2)
        pos_mse = ((pred - gtc) ** 2).mean(dim=2)
        sum_l1 += (pos_l1 * mask).sum().item()
        sum_mse += (pos_mse * mask).sum().item()
        denom += mask.sum().item()
        n_frames += gt.shape[0]
        if per_pos_l1 is None:
            per_pos_l1 = torch.zeros(m, device=gt.device)
            per_pos_cnt = torch.zeros(m, device=gt.device)
        per_pos_l1[:m] += (pos_l1 * mask).sum(dim=0)
        per_pos_cnt[:m] += mask.sum(dim=0)
        for k in buf:
            buf[k].clear()

    for fi, f in enumerate(files):
        df = pd.read_parquet(f).sort_values("frame_index").reset_index(drop=True)
        actions = np.stack(df["action"].to_numpy())
        states = np.stack(df["observation.state"].to_numpy())
        tasks = df["task_index"].to_numpy()
        T = len(df)
        for t in range(0, T, args.stride):
            valid_len = min(chunk, T - t)
            idx = [min(t + k, T - 1) for k in range(chunk)]
            gt_chunk = actions[idx]
            state = states[t].astype(np.float32)
            if lang_table is not None:
                emb = lang_table[int(tasks[t])].numpy()
                state = np.concatenate([state, emb]).astype(np.float32)
            buf["img"].append(decode_img(df[FYWANG_STATIC].iloc[t], args.img_size))
            buf["wrist"].append(decode_img(df[FYWANG_WRIST].iloc[t], args.img_size))
            buf["state"].append(state)
            buf["action"].append(gt_chunk)
            buf["valid"].append(valid_len)
            if len(buf["action"]) >= args.batch_size:
                flush()
        if (fi + 1) % 50 == 0:
            print(f"  ...{fi + 1}/{len(files)} | 当前 action_l1={sum_l1 / max(denom,1):.4f}")
    flush()

    result = {
        "checkpoint": args.checkpoint,
        "data_root": str(data_root),
        "n_frames": int(n_frames),
        "n_valid_steps": int(denom),
        "action_l1": sum_l1 / max(denom, 1),
        "action_mse": (sum_mse / max(denom, 1)) if sum_mse else None,
    }
    if per_pos_l1 is not None:
        safe = per_pos_cnt.clamp(min=1)
        result["per_chunk_position_l1"] = (per_pos_l1 / safe).cpu().tolist()
        result["per_chunk_position_count"] = per_pos_cnt.int().cpu().tolist()

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_chunk_position_l1", "per_chunk_position_count")},
                     indent=2, ensure_ascii=False))
    print(f"[done] 写出 {out}")

    if args.tracker == "swanlab":
        try:
            import swanlab
            run_name = args.run_name or f"eval_{Path(args.checkpoint).parent.parent.name}_on_D"
            run = swanlab.init(project=args.project, experiment_name=run_name,
                               config={"checkpoint": args.checkpoint,
                                       "data_root": str(data_root),
                                       "n_frames": result["n_frames"]})
            cnts = result.get("per_chunk_position_count", [])
            for i, v in enumerate(result.get("per_chunk_position_l1", [])):
                if i < len(cnts) and cnts[i] >= 30:
                    run.log({"D/per_pos_l1": v}, step=i)
            run.log({"D/action_l1": result["action_l1"],
                     "D/action_mse": result["action_mse"] or 0.0})
            run.finish()
            print(f"[swanlab] 已记录到 {args.project} / {run_name}")
        except Exception as e:
            print(f"[swanlab] 记录失败（不影响已写出的 JSON）: {e}")


if __name__ == "__main__":
    main()
