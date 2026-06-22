[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@lubei1022/calvin-act/runs/m5sfz43w/chart)
# 计算机视觉期末项目

本仓库为《计算机视觉》课程期末项目的共享代码仓库，包含两个独立任务：

* **题目一：基于 2D Gaussian Splatting（2DGS）与 AIGC 的多源三维资产生成及真实场景融合**
* **题目二：基于 LeRobot 的 ACT 策略跨环境泛化挑战**

## 总体仓库结构

```text
CV-Final-PJ/
├── task1/                              # 题目一：2DGS + AIGC 多源三维资产融合
│   ├── 01_object_A_pipeline_from_video_to_2dgs.py
│   ├── 02_extract_BC_triptych_assets.py
│   └── 03_final_fusion_video.py
├── task2/                              # 题目二：CALVIN × LeRobot ACT
│   ├── src/
│   ├── outputs/
│   ├── requirements.txt
│   └── README.md
├── README.md
├── .gitignore
└── LICENSE
```

---

## 题目一：基于 2DGS 与 AIGC 的多源三维资产生成及真实场景融合

### 任务目标

本任务以 Mip-NeRF 360 数据集中的 `bonsai` 场景作为真实背景，结合真实多视角重建、文本到三维生成和单图到三维生成三种方式，构建不同来源的三维资产，并将其融合到统一场景中进行渲染。

具体包括：

* 背景场景：基于 COLMAP 相机参数和二维高斯泼溅完成 `bonsai` 场景重建；
* 物体 A：基于手机环绕视频完成真实玩偶的多视角重建；
* 物体 B：基于文本提示词使用 threestudio 生成盆栽类三维资产；
* 物体 C：基于单张马克杯图像完成单图到三维生成；
* 最终在渲染输出层统一不同资产，通过前景提取、尺度调整、位置对齐和 Alpha 合成生成融合结果。

### 方法流程

```text
背景 bonsai 多视角图像 ──→ COLMAP + 2DGS ──→ 背景高斯模型与渲染结果
                                                        │
玩偶环绕视频 ──→ 抽帧 + 背景处理 ──→ COLMAP + 2DGS ──→ 物体 A
                                                        │
文本提示词 ──→ threestudio / SDS ──→ 物体 B            │
                                                        ├──→ 渲染层统一
马克杯单图 ──→ 前景提取 + 单图到三维 ──→ 物体 C          │
                                                        │
                                背景渲染 + A/B/C 渲染结果 ──→ 最终融合视频
```

### 核心代码说明

| 文件                                           | 功能                                                       |
| -------------------------------------------- | -------------------------------------------------------- |
| `01_object_A_pipeline_from_video_to_2dgs.py` | 物体 A 的真实多视角重建流程，包括视频抽帧、前景与背景处理、COLMAP 重建、2DGS 训练与渲染。     |
| `02_extract_BC_triptych_assets.py`           | 从物体 B、C 的 RGB、法向图、掩码三联可视化结果中提取前景资产，用于后续融合。               |
| `03_final_fusion_video.py`                   | 读取背景、物体 A 渲染帧和物体 B/C 资产，完成尺度调整、位置对齐与 Alpha 合成，并导出最终融合视频。 |

### 环境配置

实验主要在 Google Colab 与云端 GPU 环境中完成。核心依赖包括：

```text
Python
PyTorch
CUDA
COLMAP
OpenCV
NumPy
Pillow
FFmpeg
2D Gaussian Splatting
threestudio
```

其中，背景场景与物体 A 使用 COLMAP 和 2DGS 流程；物体 B 使用 threestudio 文本到三维流程；物体 C 使用单图到三维生成流程。

由于不同模块依赖的上游框架、CUDA 版本和预训练权重并不完全一致，运行前需要根据本地或云端环境修改脚本中的项目根目录、输入路径和输出路径。

### 数据准备

| 模块   | 输入数据                          | 主要处理                                      |
| ---- | ----------------------------- | ----------------------------------------- |
| 背景场景 | Mip-NeRF 360 的 `bonsai` 多视角图像 | 使用已有 COLMAP 相机参数，采用降采样图像进行 2DGS 训练。       |
| 物体 A | 手机拍摄的玩偶环绕视频                   | 抽取 36 帧，进行前景提取与背景处理，再用于 COLMAP 和 2DGS 重建。 |
| 物体 B | 盆栽/多肉植物文本描述                   | 使用文本到三维生成流程获得三维资产。                        |
| 物体 C | 单张马克杯图像                       | 前景提取、背景去除，并使用单图到三维流程恢复多视角外观。              |

原始视频帧、处理后的输入图像、训练日志、PLY 文件、转台视频和最终融合视频均未上传至 GitHub，而是统一存放在项目 Google Drive 中。

### 主要实验设置

| 模块   | 方法                          | 训练轮数       |
| ---- | --------------------------- | ---------- |
| 背景场景 | COLMAP + 2DGS               | 7000       |
| 物体 A | 视频抽帧 + 背景处理 + COLMAP + 2DGS | 3000       |
| 物体 B | threestudio 文本到三维生成         | 6000       |
| 物体 C | 单图到三维生成                     | 1000--4000 |
| 融合渲染 | 图像/视频帧层前景合成                 | --         |

### 结果与限制

真实多视角重建得到的物体 A 在主体轮廓、颜色和整体外观真实性方面表现较好，但仍受拍摄角度覆盖、绒毛细节和 COLMAP 位姿估计质量影响。物体 B 能够在没有真实图像输入的情况下生成盆栽类虚拟资产，但局部几何和纹理稳定性相对有限。物体 C 能较好保持输入图像中的杯身颜色和正面轮廓，但侧面与背面区域仍受单视角信息不足的限制。

最终融合采用渲染输出层统一，而非高斯参数级拼接。该方式实现稳定，但难以严格处理不同资产之间的真实三维遮挡、接触阴影和光照一致性。

### 结果文件与模型资产

由于视频、PLY、原始图像序列和训练输出体积较大，未直接提交至 GitHub。相关文件包括：

```text
- 最终融合视频
- 物体 B 与物体 C 的转台视频
- 物体 A 的 Gaussian PLY、mesh PLY 与 COLMAP 稀疏点云
- 物体 A 的原始帧与背景处理后帧
- 训练日志、配置文件与中间可视化结果
```

Google Drive 结果链接：**待补充**

---

## 题目二：CALVIN × LeRobot ACT —— 跨环境泛化

用 LeRobot 框架内置的 ACT（Action Chunking Transformer）算法,在 CALVIN 数据集上研究
动作策略的跨环境泛化:**基础模型**仅用环境 B 训练,**联合模型**用环境 A+B+C 混合训练,
二者在**完全未见过的环境 D** 上做 zero-shot 对比。

## 仓库结构

```
task2/
├── src/
│   ├── train.py            # 训练脚本(基础模型B/联合模型A+B+C共用)
│   ├── eval_d.py           # 在环境D上做zero-shot评测,算 Action L1
│   ├── eval_perchunk.py    # 拓展实验:算动作块内逐时间步 L1(动作分块鲁棒性)
│   ├── compare_on_D.py     # 读评测JSON,出对比表与图(柱状图 / 逐步误差曲线)
│   └── fix_stats_count.py  # 数据准备:给v2.1统计补 count 字段
│   
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

## 拓展实验

### 动作分块逐时间步误差曲线
统计ACT预测的动作块内每个时间步(第$0\sim99$步)的$L_1$误差,直接展示动作分块在
跨环境视觉偏移下的鲁棒性。对两个模型各跑一次,再叠加画曲线:

```bash
export HF_HOME=$(pwd)/hf_home
D=task2/data/calvin_split

HF_HUB_OFFLINE=1 python task2/src/eval_perchunk.py \
  --checkpoint task2/outputs/act_B/checkpoints/best \
  --train-data $D/splitB --eval-data $D/splitD --max-episodes 300 \
  --out task2/outputs/eval/act_B_perchunk_D.json

HF_HUB_OFFLINE=1 python task2/src/eval_perchunk.py \
  --checkpoint task2/outputs/act_ABC/checkpoints/best \
  --train-data $D/splitA $D/splitB $D/splitC --eval-data $D/splitD --max-episodes 300 \
  --out task2/outputs/eval/act_ABC_perchunk_D.json

python task2/src/compare_on_D.py \
  --results task2/outputs/eval/act_B_perchunk_D.json task2/outputs/eval/act_ABC_perchunk_D.json \
  --labels "B-only" "A+B+C" --out-dir task2/outputs/eval   # 生成 per_chunk_position_l1.png
```

### 跨环境退化（视觉分布偏移量化）
把基础模型(仅B)分别在A/B/C/D上评测,观察误差随视觉偏移上升:

```bash
for E in splitA splitB splitC splitD; do
  HF_HUB_OFFLINE=1 python task2/src/eval_d.py \
    --checkpoint task2/outputs/act_B/checkpoints/best \
    --train-data $D/splitB --eval-data $D/$E --max-episodes 300 \
    --out task2/outputs/eval/act_B_on_${E}.json
done
```

## 实验结果

| 模型 | 训练数据 | 环境 D 上 Action $L_1$ ↓ |
|---|---|---|
| 基础模型 | 仅环境 B | 0.2217 |
| 联合模型 | 环境 A+B+C | **0.1981** |

多环境联合训练显著提升了对未见环境的泛化(动作误差相对下降约 10.6%)。详见实验报告。

