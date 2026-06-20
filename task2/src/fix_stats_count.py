#!/usr/bin/env python
"""数据准备:给 CALVIN(xiaoma26)v2.1 数据的 episodes_stats.jsonl 补上缺失的 count 字段。
"""
import json
import sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "task2/data/calvin_split")

for sp in ["splitA", "splitB", "splitC", "splitD"]:
    md = base / sp / "meta"
    stats_path = md / "episodes_stats.jsonl"
    if not stats_path.exists():
        print(f"{sp}: 跳过(无 episodes_stats.jsonl,可能已是 v3.0)")
        continue

    # episode_index -> 帧数
    lengths = {}
    for line in open(md / "episodes.jsonl"):
        line = line.strip()
        if line:
            d = json.loads(line)
            lengths[d["episode_index"]] = d["length"]

    out = []
    for line in open(stats_path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n = lengths.get(d["episode_index"], 1)
        stats = d.get("stats", d)
        for feat, st in stats.items():
            if isinstance(st, dict) and "count" not in st:
                st["count"] = [n]
        out.append(d)

    with open(stats_path, "w") as f:
        for d in out:
            f.write(json.dumps(d) + "\n")
    print(f"{sp}: 已为 {len(out)} 个 episode 补上 count")
