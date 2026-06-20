# 题目二：CALVIN × LeRobot ACT —— 跨环境泛化

用 LeRobot 框架内置的 ACT（Action Chunking Transformer）算法,在 CALVIN 数据集上研究
动作策略的跨环境泛化:**基础模型**仅用环境 B 训练,**联合模型**用环境 A+B+C 混合训练,
二者在**完全未见过的环境 D** 上做 zero-shot 对比。

## 仓库结构

```
task2/
├── src/
│   ├── train.py            # 训练脚本(基础模型 B / 联合模型 A+B+C 共用)
│   ├── eval_d.py           # 在环境 D 上做 zero-shot 评测,算 Action L1
│   ├── compare_on_D.py     # 读两个评测 JSON,出对比表与柱状图
│   └── fix_stats_count.py  # 数据准备:给 v2.1 统计补 count 字段
└── outputs/
    ├── act_B/  act_ABC/    # 各模型的 checkpoints/best 与 train_config.json
    └── eval/               # act_B_on_D.json, act_ABC_on_D.json, 对比图
requirements.txt
README.md
```

## 环境配置

- 系统:Linux(Ubuntu);GPU:NVIDIA RTX 3090(24GB),CUDA 12.x;Python 3.12
- 关键依赖:**LeRobot 0.4.4**、PyTorch(CUDA 版)、`huggingface_hub<1.0`、SwanLab、datasets、pyarrow

```bash
conda create -n cv_env python=3.10 -y
conda activate cv_env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt`:
```
lerobot==0.4.4
huggingface_hub<1.0
swanlab
datasets
pyarrow
matplotlib
```


## 数据准备

下载已按环境拆分的 CALVIN(LeRobot 格式):[`xiaoma26/calvin-lerobot`](https://huggingface.co/datasets/xiaoma26/calvin-lerobot),含 `splitA/B/C/D` 四个环境。

```bash
# 下载数据集
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
for sp in splitA splitB splitC splitD; do
  until hf download xiaoma26/calvin-lerobot --repo-type dataset \
      --include "$sp/*" --local-dir task2/data/calvin_split; do
    echo "重试"; sleep 5
  done
done
```

```bash
# 给 v2.1 统计补count字段
python task2/src/fix_stats_count.py task2/data/calvin_split
```

```bash
# 把数据从LeRobot v2.1升级到v3.0(LeRobot 0.4.4 只支持 v3.0)
export HF_HUB_OFFLINE=0
for sp in splitA splitB splitC splitD; do
  python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
      --repo-id "$sp" --root task2/data/calvin_split --push-to-hub false
done
rm -rf task2/data/calvin_split/*_old
```


## 训练

两个模型使用**完全相同的网络结构与超参数**,唯一差异是训练数据。

```bash
export HF_HOME=$(pwd)/hf_home 

# 基础模型(仅环境B)
HF_HUB_OFFLINE=1 python task2/src/train.py \
  --data task2/data/calvin_split/splitB \
  --output-dir task2/outputs/act_B \
  --steps 20000 --batch-size 32 --num-workers 6 --max-episodes 2000 \
  --tracker swanlab --project calvin-act --run-name act_B

# 联合模型(A+B+C)
HF_HUB_OFFLINE=1 python task2/src/train.py \
  --data task2/data/calvin_split/splitA task2/data/calvin_split/splitB task2/data/calvin_split/splitC \
  --output-dir task2/outputs/act_ABC \
  --steps 20000 --batch-size 32 --num-workers 6 --max-episodes 2000 \
  --tracker swanlab --project calvin-act --run-name act_ABC
```

- 训练自动从数据中留出5%episode 作验证集,按验证Action L1选出**最优权重**,保存到
  `task2/outputs/<run>/checkpoints/best`(即提交的最优模型)。
- 超参数会存入 `task2/outputs/<run>/train_config.json`。
- `--max-episodes 2000`:每个 split 取前2000个 episode。

## 评测

在未见环境D(splitD)上对两个模型各跑一次,算Action L1,再出对比图。

```bash
export HF_HOME=$(pwd)/hf_home
D=task2/data/calvin_split

# 基础模型在D上
HF_HUB_OFFLINE=1 python task2/src/eval_d.py \
  --checkpoint task2/outputs/act_B/checkpoints/best \
  --train-data $D/splitB \
  --eval-data  $D/splitD \
  --max-episodes 300 \
  --out task2/outputs/eval/act_B_on_D.json

# 联合模型在D上
HF_HUB_OFFLINE=1 python task2/src/eval_d.py \
  --checkpoint task2/outputs/act_ABC/checkpoints/best \
  --train-data $D/splitA $D/splitB $D/splitC \
  --eval-data  $D/splitD \
  --max-episodes 300 \
  --out task2/outputs/eval/act_ABC_on_D.json

# 出对比表与柱状图
python task2/src/compare_on_D.py \
  --results task2/outputs/eval/act_B_on_D.json task2/outputs/eval/act_ABC_on_D.json \
  --labels "B-only" "A+B+C" --out-dir task2/outputs/eval
```

## 实验结果

| 模型 | 训练数据 | 环境 D 上 Action $L_1$ ↓ |
|---|---|---|
| 基础模型 | 仅环境 B | 0.2217 |
| 联合模型 | 环境 A+B+C | **0.1981** |

多环境联合训练显著提升了对未见环境的泛化(动作误差相对下降约 10.6%)。详见实验报告。

