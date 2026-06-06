"""下载 CALVIN (LeRobot 格式) 数据集：huiwon/calvin_task_ABC_D

这份数据已经是 LeRobot 格式（含 data/ meta/ videos/），全部约 7.81 GB，
所以无需再写 CALVIN->LeRobot 转换脚本。

国内下载慢的解决办法（重要）：
  直连 huggingface.co 在国内通常很慢。用镜像站 hf-mirror.com 可大幅提速。
  本脚本加 --mirror 即自动设置 HF_ENDPOINT=https://hf-mirror.com。

依赖：
    pip install -U huggingface_hub
    pip install -U hf_transfer     # 可选，配合 --fast 进一步提速

用法示例
--------
全部下载（推荐在 GPU 服务器/AutoDL 上跑，约 7.81 GB）：
    python data/download_calvin.py --mirror --fast

只下第 0 个分片（先小规模验证流程，量力而行）：
    python data/download_calvin.py --mirror --shards 0

下到自定义相对目录：
    python data/download_calvin.py --mirror --out ./data/calvin_raw
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "huiwon/calvin_task_ABC_D"
N_SHARDS = 4  # 分片命名：calvin_task_ABC_D_lerobot_{i}_4 , i=0..3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download huiwon/calvin_task_ABC_D (LeRobot format).")
    p.add_argument("--out", type=str, default="./data/calvin_raw",
                   help="下载到的本地目录（相对路径）")
    p.add_argument("--shards", type=int, nargs="*", default=None,
                   help="只下指定分片编号(0-3)，如 --shards 0 1；不填则全下")
    p.add_argument("--mirror", action="store_true",
                   help="使用 hf-mirror.com 镜像加速（国内强烈建议）")
    p.add_argument("--endpoint", type=str, default="https://hf-mirror.com",
                   help="自定义镜像地址，配合 --mirror 使用")
    p.add_argument("--fast", action="store_true",
                   help="启用 Xet 高速下载")
    p.add_argument("--no-xet", action="store_true",
                   help="禁用 Xet 协议（镜像不支持 Xet 时用）")
    p.add_argument("--workers", type=int, default=8, help="并发下载线程数")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 镜像与加速：必须在 import huggingface_hub 前设置环境变量
    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.endpoint
        print(f"[mirror] HF_ENDPOINT = {args.endpoint}")
    if args.fast:
        # 新版 huggingface_hub 用 Xet 高速传输（旧的 HF_HUB_ENABLE_HF_TRANSFER 已废弃）
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        print("[fast] Xet 高速传输已开启（若遇 Xet 报错，改设 HF_HUB_DISABLE_XET=1 再下）")
    # 镜像有时不支持 Xet：允许通过 --no-xet 关闭
    if getattr(args, "no_xet", False):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("[xet] 已禁用 Xet 协议")

    from huggingface_hub import snapshot_download

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 选择性下载：只取指定分片的文件
    allow_patterns = None
    if args.shards:
        for s in args.shards:
            if s < 0 or s >= N_SHARDS:
                raise ValueError(f"分片编号必须在 0..{N_SHARDS - 1} 之间，收到 {s}")
        allow_patterns = [f"calvin_task_ABC_D_lerobot_{s}_4/*" for s in args.shards]
        print(f"[shards] 仅下载: {allow_patterns}")
    else:
        print("[shards] 下载全部 4 个分片（约 7.81 GB）")

    print(f"[download] repo={REPO_ID} -> {out_dir}")
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=allow_patterns,
        max_workers=args.workers,
        # snapshot_download 默认断点续传，中断后重跑本命令即可继续
    )
    print(f"[done] 数据已下载到: {local_path}")
    print("下一步：用 inspect/split 脚本确定每个 episode 属于环境 A/B/C/D，"
          "再分出'环境 B'与'A+B+C'两份训练集。")


if __name__ == "__main__":
    main()
