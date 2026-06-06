# CALVIN × LeRobot ACT —— 跨环境泛化（题目二）

用 LeRobot 内置的 ACT 算法，在 CALVIN 上研究动作策略的跨环境泛化：
基础模型只用环境 B 训练，联合模型用 A+B+C 训练，二者在未见过的环境 D 上做
zero-shot 对比。

## 目录结构

```
data/
  download_calvin.py      # 下载 huiwon/calvin_task_ABC_D（LeRobot 格式，~7.81GB）
  fetch_scene_info.py     # 只从远程 zip 抽 scene_info.npy（拆分环境用）
  split_by_env.py         # 按 original_frame_idx + scene_info 把 episode 分到 A/B/C
src/
  utils.py                # 设备(cuda>mps>cpu)/种子/batch 搬运
  calvin_data.py          # 跨分片按环境过滤 + 语言条件 的数据集
  train_base.py           # 训练脚本（B 与 A+B+C 共用，只换清单）
  eval_action_error.py    # 环境 D 上的离线动作误差评测
environment.yml           # conda 隔离环境
```

## 环境配置

```bash
conda env create -f environment.yml
conda activate calvin-act
# PyTorch 按平台单独装：
#   GPU 服务器: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   Mac:        pip install torch torchvision
pip show lerobot   # 记下版本号写进报告
```

## 数据准备

```bash
# 1) 下载（国内用镜像加速）
python data/download_calvin.py --mirror --fast
python data/split_by_env.py describe --root ./data/calvin_raw   # 验证完整性

# 2) 按环境拆分，得到 split_B.json / split_ABC.json
#    默认用内置的官方 ABC_D 场景边界（已用数据校验），无需额外文件：
python data/split_by_env.py split --root ./data/calvin_raw --out ./data/env_splits
#    若拿到官方 scene_info.npy，可覆盖（更权威）：
#    python data/split_by_env.py split --root ./data/calvin_raw \
#        --scene-info ./data/calvin_meta/scene_info.npy --out ./data/env_splits
```

## 训练（两个模型，超参数完全一致，只换清单）

```bash
# 基础模型（仅环境 B）
python src/train_base.py \
    --shards-root ./data/calvin_raw \
    --split-manifest ./data/env_splits/split_B.json \
    --output-dir ./outputs/act_B \
    --steps 100000 --batch-size 32 --tracker swanlab --run-name act_B

# 联合模型（A+B+C）
python src/train_base.py \
    --shards-root ./data/calvin_raw \
    --split-manifest ./data/env_splits/split_ABC.json \
    --output-dir ./outputs/act_ABC \
    --steps 100000 --batch-size 32 --tracker swanlab --run-name act_ABC
```

> 训练默认从清单留出 5% episode 作验证集（`--val-frac`），每 `--eval-freq` 步评一次验证
> Action L1，**最优权重存到 `checkpoints/best`**——这就是提交给老师的"最优模型"文件。

## 评测（环境 D，zero-shot）

```bash
python src/eval_action_error.py --checkpoint ./outputs/act_B/checkpoints/best   \
    --shards-root ./data/calvin_D --split-manifest ./data/env_splits/split_D.json \
    --out ./outputs/eval/act_B_on_D.json
python src/eval_action_error.py --checkpoint ./outputs/act_ABC/checkpoints/best \
    --shards-root ./data/calvin_D --split-manifest ./data/env_splits/split_D.json \
    --out ./outputs/eval/act_ABC_on_D.json
```

## 待确认 / 已知问题

- **环境 D 测试数据来源**：huiwon 的 ABC_D 只含训练用的 A/B/C，不含 D。需向老师确认
  D 的测试数据是否单独提供，或改用 CALVIN 仿真器跑 rollout。
- **scene_info.npy**：早期版本有 bug，请用修正版（最稳是直接问老师要）。
- 代码已通过语法检查；标了 `# VERIFY` 的几处（LeRobot 单样本字段名、动作块预测接口、
  权重加载接口）需在装好 LeRobot 的机器上首跑时确认一次。

## 外部链接

- GitHub 仓库：<在此填写>
- 模型权重网盘：<在此填写（含提取码）>
