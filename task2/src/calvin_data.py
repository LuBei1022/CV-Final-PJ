"""跨分片、按环境过滤的 CALVIN 数据加载（含语言条件）。

为什么需要它
------------
- 数据是 4 个独立的 LeRobot 分片，而"环境 B"/"A+B+C"的 episode 散落在所有分片里
  （见 data/split_by_env.py 产出的 split_*.json 清单）。这里把多个分片按清单
  过滤后拼成一个可训练的数据集。
- 语言条件：CALVIN 是多任务数据，同一画面在不同任务下动作不同。若不给任务信息，
  ACT 会学到互相冲突的目标。我们用冻结文本编码器把任务指令编码成向量，拼接到
  observation.state 末尾——这样 stock LeRobot ACT 的 state 投影层自动消化它，
  无需改 ACT 源码。

注意（需在有数据/装好 LeRobot 的机器上确认的点，已在代码中标注 # VERIFY）：
- LeRobotDataset 单样本里 task_index 字段名与类型；
- 样本中 observation.state 是否为单帧 (state_dim,)（本模块按此假设拼接）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


# ------------------------------ 清单 / 任务 ------------------------------
def load_manifest(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def group_by_shard(manifest: list[dict]) -> dict[str, list[int]]:
    """{shard_name: [episode_index, ...]}（去重排序）。"""
    g: dict[str, set] = {}
    for e in manifest:
        g.setdefault(e["shard"], set()).add(int(e["episode_index"]))
    return {k: sorted(v) for k, v in g.items()}


def load_task_map(shard_dir: Path) -> dict[int, str]:
    """读 tasks 表 -> {task_index: 指令文本}。兼容 v2.1(jsonl) 与 v3.0(parquet)。"""
    shard_dir = Path(shard_dir)

    jl = shard_dir / "meta" / "tasks.jsonl"
    if jl.exists():
        m: dict[int, str] = {}
        for line in jl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            m[int(d["task_index"])] = d["task"]
        return m

    import pandas as pd
    pqs = sorted(shard_dir.glob("meta/tasks/**/*.parquet")) or \
          sorted(shard_dir.glob("meta/tasks*.parquet"))
    if not pqs:
        raise FileNotFoundError(
            f"{shard_dir}/meta 下既没有 tasks.jsonl 也没有 tasks parquet，数据可能不完整")
    df = pd.concat([pd.read_parquet(p) for p in pqs]).reset_index()
    cols = list(df.columns)
    idx_col = next((c for c in cols if "task_index" in str(c).lower()), None)
    if idx_col is None:
        idx_col = next(c for c in cols if pd.api.types.is_integer_dtype(df[c]))
    txt_col = next((c for c in cols if c != idx_col and df[c].dtype == object), None)
    if txt_col is None:
        txt_col = next(c for c in cols if c != idx_col)
    return {int(i): str(t) for i, t in zip(df[idx_col], df[txt_col])}


# ------------------------------ 语言编码器 ------------------------------
class LangEncoder:
    """把任务指令文本编码成固定维度向量（L2 归一化，可截断维度）。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", dim: int | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = dim  # None=用模型原始维度(MiniLM=384)；否则截断到前 dim 维

    @property
    def out_dim(self) -> int:
        base = self.model.get_sentence_embedding_dimension()
        return base if self.dim is None else min(self.dim, base)

    def table_for(self, task_map: dict[int, str]) -> dict[int, torch.Tensor]:
        """{task_index: tensor(out_dim)}。"""
        idxs = sorted(task_map)
        texts = [task_map[i] for i in idxs]
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if self.dim is not None:
            embs = embs[:, : self.dim]
            # 截断后再 L2 归一化，保持单位长度
            norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            embs = embs / norm
        return {i: torch.tensor(e, dtype=torch.float32) for i, e in zip(idxs, embs)}


# ------------------------------ 数据集 ------------------------------
class EnvFilteredDataset(Dataset):
    """把多个分片按清单过滤后拼接；可选把语言向量拼进 observation.state。"""

    def __init__(self, shards_root: str | Path, manifest: list[dict],
                 delta_timestamps: dict, lang_encoder: LangEncoder | None = None):
        self.shards_root = Path(shards_root)
        self.lang_encoder = lang_encoder
        self.subsets: list[LeRobotDataset] = []
        self.tables: list[dict[int, torch.Tensor] | None] = []
        self._cumlen: list[int] = []

        groups = group_by_shard(manifest)
        if not groups:
            raise ValueError("清单为空，请检查 split_*.json。")

        total = 0
        for shard_name, eps in groups.items():
            shard_dir = self.shards_root / shard_name
            ds = LeRobotDataset(shard_name, root=shard_dir, episodes=eps,
                                delta_timestamps=delta_timestamps,
                                tolerance_s=1e-3)
            self.subsets.append(ds)
            self.tables.append(
                lang_encoder.table_for(load_task_map(shard_dir)) if lang_encoder else None
            )
            total += len(ds)
            self._cumlen.append(total)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def _locate(self, i: int) -> tuple[int, int]:
        """全局索引 -> (子集编号, 子集内局部索引)。"""
        for j, c in enumerate(self._cumlen):
            if i < c:
                prev = self._cumlen[j - 1] if j > 0 else 0
                return j, i - prev
        raise IndexError(i)

    def __getitem__(self, i: int) -> dict:
        j, local = self._locate(i)
        item = self.subsets[j][local]
        table = self.tables[j]
        if table is not None:
            ti = item["task_index"]
            ti = int(ti.item()) if torch.is_tensor(ti) else int(ti)  # VERIFY 字段名/类型
            emb = table[ti]
            state = item["observation.state"]
            # 假设 state 为单帧 (state_dim,)；与 emb 在最后一维拼接
            item["observation.state"] = torch.cat([state, emb], dim=-1)
        return item


# ------------------------------ 特征 / 统计 增广 ------------------------------
def augment_state_feature(input_features: dict, emb_dim: int):
    """把 observation.state 的特征维度扩大 emb_dim（用于 ACTConfig.input_features）。"""
    import dataclasses
    feat = input_features["observation.state"]
    old = feat.shape
    new_shape = (old[0] + emb_dim,) + tuple(old[1:])
    try:
        input_features["observation.state"] = dataclasses.replace(feat, shape=new_shape)
    except TypeError:
        feat.shape = new_shape  # 退路：直接改属性
    return input_features


def augment_state_stats(stats: dict, emb_dim: int) -> dict:
    """把 observation.state 的归一化统计扩展 emb_dim 维。

    语言向量已 L2 归一化，故对这些维不做标准化：mean=0, std=1, min=-1, max=1。
    """
    import copy
    if emb_dim <= 0:
        return stats
    stats = copy.deepcopy(stats)
    s = stats["observation.state"]
    pad = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0}
    for key, fill in pad.items():
        if key in s and s[key] is not None:
            arr = np.asarray(s[key], dtype=np.float32).reshape(-1)
            s[key] = np.concatenate([arr, np.full(emb_dim, fill, dtype=np.float32)])
    return stats


def first_shard_metadata(shards_root: str | Path, manifest: list[dict]) -> LeRobotDatasetMetadata:
    """取清单里第一个分片的元信息（fps / features / stats 的代表）。"""
    shard_name = next(iter(group_by_shard(manifest)))
    shard_dir = Path(shards_root) / shard_name
    return LeRobotDatasetMetadata(shard_name, root=shard_dir)


# ACT 实际需要的特征白名单（丢弃 annotation/original_frame_idx/index 等多余列）
POLICY_FEATURE_WHITELIST = {
    "action",
    "observation.state",
    "observation.images.image",
    "observation.images.wrist_image",
}


def filter_policy_features(features: dict) -> dict:
    """只保留 ACT 需要的特征。"""
    return {k: v for k, v in features.items() if k in POLICY_FEATURE_WHITELIST}
