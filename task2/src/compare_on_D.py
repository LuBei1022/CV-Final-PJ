"""要素 3 收尾：对比两个模型在环境 D 上的 zero-shot 表现，并出报告用图表。

输入是 eval_action_error.py 产出的若干 JSON（通常是 act_B 与 act_ABC 各一份）。
输出：
  - comparison.md / comparison.csv   并排对比表（整体 Action L1 / MSE）
  - overall_l1.png                   整体动作误差柱状图
  - per_chunk_position_l1.png        按 chunk 内时间步分解的 L1 曲线
                                     （用于分析 ACT 动作分块在视觉偏移下的鲁棒性）

用法：
    python src/compare_on_D.py \
        --results ./outputs/eval/act_B_on_D.json ./outputs/eval/act_ABC_on_D.json \
        --labels  "B-only" "A+B+C" \
        --out-dir ./outputs/eval

依赖：matplotlib
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境（服务器）也能出图
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Compare models' zero-shot results on env D.")
    p.add_argument("--results", nargs="+", required=True, help="eval 输出的 JSON 路径（≥2 个）")
    p.add_argument("--labels", nargs="+", default=None, help="每个结果的图例名，缺省用文件名")
    p.add_argument("--out-dir", default="./outputs/eval")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [json.loads(Path(p).read_text()) for p in args.results]
    labels = args.labels or [Path(p).stem for p in args.results]
    if len(labels) != len(results):
        raise SystemExit("--labels 数量需与 --results 一致")

    # ---------------- 对比表 ----------------
    rows = []
    for lab, r in zip(labels, results):
        rows.append({
            "model": lab,
            "n_frames": r.get("n_frames"),
            "action_l1": r.get("action_l1"),
            "action_mse": r.get("action_mse"),
        })

    # markdown
    md = ["| 模型 | 评测帧数 | Action L1 ↓ | Action MSE ↓ |",
          "|---|---|---|---|"]
    for r in rows:
        mse = f"{r['action_mse']:.4f}" if r["action_mse"] is not None else "-"
        md.append(f"| {r['model']} | {r['n_frames']} | {r['action_l1']:.4f} | {mse} |")
    (out_dir / "comparison.md").write_text("\n".join(md))

    # csv
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n_frames", "action_l1", "action_mse"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))

    # ---------------- 整体 L1 柱状图 ----------------
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [r["action_l1"] for r in rows]
    bars = ax.bar(labels, vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(labels)])
    ax.set_ylabel("Action L1 on env D (lower is better)")
    ax.set_title("Zero-shot action error on unseen env D")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "overall_l1.png", dpi=150)
    plt.close(fig)

    # ---------------- 按 chunk 内时间步的 L1 曲线 ----------------
    has_curve = any("per_chunk_position_l1" in r for r in results)
    if has_curve:
        fig, ax = plt.subplots(figsize=(6, 4))
        for lab, r in zip(labels, results):
            curve = r.get("per_chunk_position_l1")
            if curve:
                ax.plot(range(len(curve)), curve, marker=".", label=lab)
        ax.set_xlabel("Action chunk position (timestep)")
        ax.set_ylabel("L1 error (lower is better)")
        ax.set_title("Per-position L1 within action chunk (robustness to visual shift)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "per_chunk_position_l1.png", dpi=150)
        plt.close(fig)
        print(f"[plot] per_chunk_position_l1.png 已保存")
    else:
        print("[plot] 结果里没有 per_chunk_position_l1，跳过曲线图")

    print(f"[done] 图表与对比表已写入 {out_dir}")


if __name__ == "__main__":
    main()
