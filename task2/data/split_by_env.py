"""把 calvin_task_ABC_D（LeRobot 格式）按环境 A/B/C 拆分。

背景
----
这份数据的 4 个分片是"平均切块上传"，不是按环境分的。但每帧都带一个
`original_frame_idx`——它是原始 CALVIN task_ABC_D 训练数据里的全局帧号。
原始 CALVIN 用 `scene_info.npy` 把帧号区间映射到场景，形如：
    {'calvin_scene_B': [0, 598909],
     'calvin_scene_C': [598910, 1191338],
     'calvin_scene_A': [1191339, 1795044]}
（具体数值以你拿到的 scene_info.npy 为准——早期版本有 bug，请用修正版。）

于是：读每条 episode 的 original_frame_idx，落在哪个区间 → 属于哪个环境。

用法
----
1) 先看分布（不需要 scene_info，用来确认 original_frame_idx 合理、找下载是否完整）：
    python data/split_by_env.py describe --root ./data/calvin_raw

2) 正式拆分（需要 scene_info.npy）：
    python data/split_by_env.py split \
        --root ./data/calvin_raw \
        --scene-info ./data/calvin_meta/scene_info.npy \
        --out ./data/env_splits

输出
----
./data/env_splits/episodes_by_env.json   # 每个环境的 (分片, episode_index) 全清单
./data/env_splits/split_B.json           # 训练"基础模型"用：仅环境 B
./data/env_splits/split_ABC.json         # 训练"联合模型"用：A+B+C 全部

依赖：pip install pandas pyarrow numpy
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def find_shards(root: Path) -> list[Path]:
    """返回所有分片目录（含 data/ 的）。"""
    shards = sorted(p for p in root.glob("calvin_task_ABC_D_lerobot_*") if (p / "data").is_dir())
    if not shards:
        # 也许 root 本身就是单个分片
        if (root / "data").is_dir():
            shards = [root]
    return shards


def iter_episodes(shard: Path):
    """逐个返回 (episode_index, parquet_path)。"""
    for pq in sorted(shard.glob("data/chunk-*/episode_*.parquet")):
        m = EPISODE_RE.search(pq.name)
        if m:
            yield int(m.group(1)), pq


def episode_frame_idx(pq: Path) -> int:
    """取该 episode 的代表帧号（用中位数，避免边界帧误判）。"""
    df = pd.read_parquet(pq, columns=["original_frame_idx"])
    return int(np.median(df["original_frame_idx"].to_numpy()))


def load_scene_info(path: Path) -> dict[str, tuple[int, int]]:
    raw = np.load(path, allow_pickle=True).item()
    ranges = {}
    for key, val in raw.items():
        env = key.split("_")[-1].upper()  # calvin_scene_B -> B
        ranges[env] = (int(val[0]), int(val[1]))
    return ranges


# CALVIN task_ABC_D 官方场景边界（原始全局帧号区间）。
# 已用实际数据校验：所有 episode 均能归类、无落空。若拿到官方 scene_info.npy，
# 用 --scene-info 覆盖即可（更权威）。
BUILTIN_ABC_D = {
    "B": (0, 598909),
    "C": (598910, 1191338),
    "A": (1191339, 1795044),
}


def env_of(frame_idx: int, ranges: dict[str, tuple[int, int]]) -> str | None:
    for env, (start, end) in ranges.items():
        if start <= frame_idx <= end:
            return env
    return None


# ------------------------------ describe ------------------------------
def cmd_describe(args) -> None:
    root = Path(args.root).resolve()
    shards = find_shards(root)
    print(f"[describe] root={root}")
    print(f"[describe] 找到分片: {[s.name for s in shards] or '无'}")
    all_idx = []
    for shard in shards:
        eps = list(iter_episodes(shard))
        idxs = [episode_frame_idx(pq) for _, pq in eps]
        all_idx.extend(idxs)
        if idxs:
            print(f"  {shard.name}: {len(eps)} episodes | "
                  f"original_frame_idx {min(idxs)} ~ {max(idxs)}")
        # 完整性检查
        n_video = len(list(shard.glob("videos/**/*.mp4")))
        n_meta = (shard / "meta").is_dir()
        print(f"     videos mp4: {n_video} | meta/: {'有' if n_meta else '缺失'}")
    if all_idx:
        a = np.array(all_idx)
        print(f"[describe] 全部 episode 数: {len(a)} | "
              f"帧号范围 {a.min()} ~ {a.max()}")
        print("[describe] 提示：若要拆分，请提供 scene_info.npy 后运行 split 子命令。")


# ------------------------------- split --------------------------------
def cmd_split(args) -> None:
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.scene_info:
        ranges = load_scene_info(Path(args.scene_info))
        print(f"[split] 用 scene_info.npy 区间: {ranges}")
    else:
        ranges = dict(BUILTIN_ABC_D)
        print(f"[split] 用内置 ABC_D 官方边界: {ranges}（如有官方 scene_info.npy 可用 --scene-info 覆盖）")

    shards = find_shards(root)
    by_env: dict[str, list] = defaultdict(list)
    n_unknown = 0
    for shard in shards:
        for ep_idx, pq in iter_episodes(shard):
            fidx = episode_frame_idx(pq)
            env = env_of(fidx, ranges)
            if env is None:
                n_unknown += 1
                continue
            by_env[env].append({"shard": shard.name, "episode_index": ep_idx, "frame_idx": fidx})

    # 统计
    print("[split] 每环境 episode 数:")
    for env in sorted(by_env):
        print(f"   环境 {env}: {len(by_env[env])}")
    if n_unknown:
        print(f"[split] ⚠️ {n_unknown} 条 episode 未落入任何区间，"
              f"可能 scene_info 与本数据不匹配，请核对。")

    # 落盘：完整清单 + 两个训练用清单
    (out / "episodes_by_env.json").write_text(
        json.dumps(by_env, indent=2, ensure_ascii=False))

    split_B = by_env.get("B", [])
    split_ABC = by_env.get("A", []) + by_env.get("B", []) + by_env.get("C", [])
    (out / "split_B.json").write_text(json.dumps(split_B, indent=2, ensure_ascii=False))
    (out / "split_ABC.json").write_text(json.dumps(split_ABC, indent=2, ensure_ascii=False))
    print(f"[split] 已写出: {out}/episodes_by_env.json, split_B.json ({len(split_B)}), "
          f"split_ABC.json ({len(split_ABC)})")


def main() -> None:
    p = argparse.ArgumentParser(description="Split CALVIN ABC_D (LeRobot) by environment.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("describe", help="查看分布与下载完整性，不需要 scene_info")
    d.add_argument("--root", default="./data/calvin_raw")
    d.set_defaults(func=cmd_describe)

    s = sub.add_parser("split", help="按环境拆分（默认用内置官方边界，可选 scene_info.npy）")
    s.add_argument("--root", default="./data/calvin_raw")
    s.add_argument("--scene-info", default=None,
                   help="可选：CALVIN scene_info.npy 路径；不填则用内置 ABC_D 官方边界")
    s.add_argument("--out", default="./data/env_splits")
    s.set_defaults(func=cmd_split)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
