# Value Function 训练框架文档

## 📚 概述

本项目实现了一个基于 **Qwen-VL** 编码器的价值函数（Value Function）训练框架，用于机器人视觉-语言-动作（VLA）模型的离线强化学习。该框架参考了 **π₀.₆\*** 论文中的价值函数训练方法，采用分布式的 bin 离散化策略，将连续的价值预测问题转化为分类问题。

## 📖 参考论文

### π₀.₆\*: Scaling Vision-Language-Action Models with RL

- **论文链接**: https://arxiv.org/pdf/2511.14759
- **核心思想**: 
  - 使用离线强化学习训练视觉-语言-动作（VLA）模型
  - 通过优势条件策略（Advantage-conditioned Policies）进行策略学习
  - 价值函数采用 **201-bin 离散化**方法（参考 C51/Distributional RL）

### 关键方法

1. **Reward 定义**（π₀.₆\* 风格）:
   - 非最后一步: `r_t = -1`
   - 最后一步且成功: `r_T-1 = 0`
   - 最后一步且失败: `r_T-1 = -big_negative` (默认 -100)

2. **Return 计算**:
   - 使用折扣因子 `gamma = 1.0`（无折扣）
   - Return: `V_hat[t] = sum_{k=t}^{T-1} gamma^{k-t} * r[k]`

3. **Bin 离散化**（参考 C51）:
   - 将连续 return 值离散化为 **201 个 bin**
   - 使用交叉熵损失进行训练
   - 推理时通过期望值计算连续 value: `V = sum(probs * bin_values)`

## 🏗️ 框架架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Value Function Training                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Dataset    │───▶│ Value Target     │───▶│  QwenValue   │
│  (LeRobot)   │    │  Wrapper         │    │   Model      │
└──────────────┘    └──────────────────┘    └──────────────┘
        │                     │                     │
        │                     │                     │
        │  Compute Returns    │  Add value_bin      │  Predict
        │  (π₀.₆* rules)     │  (201 bins)        │  (logits)
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Cross-Entropy   │
                    │      Loss        │
                    └──────────────────┘
```

## 🔧 核心组件

### 1. QwenValue 模型 (`starVLA/model/framework/QwenValue.py`)

基于 Qwen-VL 编码器的价值函数模型。

**架构**:
- **编码器**: Qwen-VL (Vision-Language Model)
- **Value Head**: 2层 MLP
  - 输入: Qwen-VL 的 `last_hidden` [B, L, H]
  - Pooling: Mean pooling → [B, H]
  - MLP: `Linear(H, H) → ReLU → Linear(H, 201)`
  - 输出: `logits` [B, 201]

**关键方法**:
- `forward()`: 训练时计算交叉熵损失
- `predict_value()`: 推理时预测连续 value（通过期望值计算）

### 2. LeRobotWithValueTarget (`starVLA/dataloader/value_targets_wrapper.py`)

数据集包装器，动态计算 value target 和 value bin。

**功能**:
1. **在线计算 Return**:
   - 根据 π₀.₆\* 规则计算每步的 reward
   - 从后往前计算 return（Monte Carlo return）

2. **Bin 离散化**:
   - 支持两种模式：
     - **固定范围**: 用户指定 `bin_min` 和 `bin_max`
     - **数据驱动**: 从采样数据中估计 min/max
   - 将连续 return 映射到 [0, 200] 的整数 bin

3. **进度显示**:
   - 实时显示 return 计算进度
   - 显示处理速度、ETA 等信息

### 3. 训练脚本 (`starVLA/training/train_value.py`)

分布式训练脚本，支持 `torchrun` 多 GPU/多节点训练。

**特性**:
- 自动检测分布式环境（torchrun）
- 支持数据混合（data mixture）
- 实时训练进度显示
- 自动保存模型检查点

### 4. Bin 范围计算工具 (`starVLA/training/compute_value_bin_range.py`)

独立脚本，用于预先计算数据集的 bin_min 和 bin_max。

**用途**:
- 大数据集时，避免在训练时重复计算
- 可以保存为 JSON 文件，供训练时使用

## 📊 数据流程

### 1. 数据加载

```python
# 使用 data mixture 加载多个数据集
mixture_spec = DATASET_NAMED_MIXTURES["sq_agi_beta"]
# 例如: [("AgiBotWorld-Beta-LeRobot", 1.0, "agibot_genie1_joint"),
#        ("suqian_agibot_lerobot_data", 1.0, "agibot_genie1_joint")]
```

### 2. Return 计算

```python
# 对于每条轨迹 (trajectory):
rewards = [-1, -1, ..., -1, 0]  # 成功: 最后一步为 0
# 或
rewards = [-1, -1, ..., -1, -100]  # 失败: 最后一步为 -100

# 从后往前计算 return
returns = [V_0, V_1, ..., V_T-1]
# 其中 V_t = r_t + gamma * V_{t+1}
```

### 3. Bin 离散化

```python
# 将连续 return 映射到 bin index
bin_delta = (bin_max - bin_min) / 200
bin_index = round((return - bin_min) / bin_delta)
bin_index = clip(bin_index, 0, 200)  # 确保在有效范围内
```

### 4. 训练

```python
# Forward pass
logits = model(observations)  # [B, 201]
targets = value_bins  # [B] (0 ~ 200)

# Loss
loss = CrossEntropyLoss(logits, targets)
```

### 5. 推理

```python
# 预测 bin 分布
probs = softmax(logits)  # [B, 201]
bin_index = argmax(probs)  # [B]

# 转换为连续 value
bin_values = [bin_min, bin_min + delta, ..., bin_max]  # [201]
value = sum(probs * bin_values)  # [B] (期望值)
```

## 🚀 使用方法

### 1. 准备配置文件

编辑 `examples/Suqian_agibot/train_files/starvla_value_function.yaml`:

```yaml
framework:
  name: QwenValue
  qwenvl:
    base_vlm: /path/to/Qwen3-VL-4B-Instruct
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
  value_num_bins: 201
```

### 2. 配置训练参数

编辑 `examples/Suqian_agibot/train_files/run_value.sh`:

```bash
# 数据配置
DATA_ROOT_DIR=/mnt/workspace/datasets
DATA_MIX=sq_agi_beta

# Bin 范围（固定模式）
BIN_MIN=-3000.0
BIN_MAX=0.0

# 训练超参
EPOCHS=1
BATCH_SIZE=8
LR=3e-5
NUM_WORKERS=4

# 输出目录
OUTPUT_DIR=./outputs_value
```

### 3. 启动训练

```bash
bash examples/Suqian_agibot/train_files/run_value.sh
```

### 4. （可选）预先计算 Bin 范围

对于大数据集，可以预先计算 bin 范围：

```bash
bash examples/Suqian_agibot/train_files/compute_bin_range.sh
```

然后在 `run_value.sh` 中使用：

```bash
BIN_RANGE_JSON=examples/Suqian_agibot/train_files/value_bin_range.json
```

## ⚙️ 配置说明

### Bin 范围配置（三种方式，按优先级）

1. **命令行直接指定**（最高优先级）:
   ```bash
   --bin_min -3000.0 --bin_max 0.0
   ```

2. **JSON 文件**:
   ```bash
   --bin_range_json value_bin_range.json
   ```

3. **数据驱动模式**（最低优先级）:
   - 不指定 bin_min/bin_max
   - 自动从采样数据中估计

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 1 |
| `--batch_size` | Batch size | 8 |
| `--learning_rate` | 学习率 | 3e-5 |
| `--num_workers` | DataLoader workers | 4 |
| `--num_bins` | Bin 数量 | 201 |
| `--gamma` | 折扣因子 | 1.0 |
| `--big_negative` | 失败奖励绝对值 | 100.0 |
| `--output_dir` | 模型保存目录 | ./outputs_value |

## 📈 训练输出

### 初始化阶段

```
[LeRobotWithValueTarget] Computing returns for all 50000 trajectories...
[LeRobotWithValueTarget] Return 计算进度: 5000/50000 (10.0%) | 已计算轨迹: 5000 | 总步数: 125000 | 速度: 45.2 traj/s | ETA: 00:16
...
```

### 训练阶段

```
开始训练: 总 epoch=1, 每 epoch 步数=1000, 总步数=1000
Batch size=8, Learning rate=3e-05
--------------------------------------------------------------------------------
[Epoch 1/1] Step 10/1000 (1.0%) | Loss: 2.3456 | Speed: 1.23 steps/s | ETA: 00:13:25
[Epoch 1/1] Step 20/1000 (2.0%) | Loss: 2.1234 | Speed: 1.25 steps/s | ETA: 00:13:02
...
Epoch 1/1 完成，耗时: 769.23秒
--------------------------------------------------------------------------------
训练完成，模型已保存到: ./outputs_value/qwen_value_final.pt
```

## 🔍 技术细节

### 1. 分布式训练

- 使用 `torchrun` 自动初始化进程组
- 支持多 GPU（单节点）和多节点训练
- 使用 `DistributedDataParallel` (DDP) 进行数据并行

### 2. Bin 离散化策略

参考 **C51** (Distributional DQN) 方法：

- **Bin 数量**: 201（与 π₀.₆\* 一致）
- **Bin 范围**: [bin_min, bin_max]，均匀分布
- **Bin delta**: `(bin_max - bin_min) / 200`
- **映射公式**: `bin_index = round((return - bin_min) / bin_delta)`

### 3. 损失函数

使用 **交叉熵损失**（分类问题）：

```python
loss = CrossEntropyLoss(logits, target_bins)
# logits: [B, 201]
# target_bins: [B] (0 ~ 200)
```

### 4. 推理时的 Value 计算

使用 **期望值**方法将离散分布转换为连续值：

```python
# 计算每个 bin 对应的 value
bin_values = [bin_min, bin_min + delta, ..., bin_max]  # [201]

# 期望值
value = sum(probs * bin_values)  # [B]
```

### 5. 超出范围处理

如果 return 值超出 [bin_min, bin_max] 范围：

- **自动裁剪**到有效范围 [0, 200]
- **记录警告**（仅第一次发生时打印，避免日志过多）

## 📁 文件结构

```
JoyRA-RL/
├── starVLA/
│   ├── model/framework/
│   │   └── QwenValue.py              # QwenValue 模型定义
│   ├── dataloader/
│   │   └── value_targets_wrapper.py  # Value target 计算包装器
│   └── training/
│       ├── train_value.py            # 训练脚本
│       └── compute_value_bin_range.py # Bin 范围计算工具
└── examples/Suqian_agibot/train_files/
    ├── run_value.sh                   # 训练启动脚本
    ├── compute_bin_range.sh           # Bin 范围计算脚本
    ├── starvla_value_function.yaml    # 配置文件
    └── README_VALUE_TRAINING.md        # 本文档
```

## 🎯 关键设计决策

1. **为什么使用 Bin 离散化？**
   - 参考 π₀.₆\* 和 C51 方法
   - 将回归问题转化为分类问题，更稳定
   - 可以捕获 value 分布的不确定性

2. **为什么使用固定 bin_min/bin_max？**
   - 确保不同数据集/训练阶段的一致性
   - 避免因数据分布变化导致需要重新训练
   - 当前设置: `bin_min=-3000, bin_max=0`

3. **为什么使用 Mean Pooling？**
   - 简单有效，与 Qwen-VL 的 hidden states 兼容
   - 可以后续替换为 CLS token 或 Attention Pooling

4. **为什么使用在线计算 Return？**
   - 原始数据集不包含 reward/return 信息
   - 根据 π₀.₆\* 规则动态计算，灵活可配置

## 🔗 相关资源

- **π₀.₆\* 论文**: https://arxiv.org/pdf/2511.14759
- **C51 (Distributional DQN)**: https://arxiv.org/abs/1707.06887
- **Qwen-VL**: https://github.com/QwenLM/Qwen-VL

## 📝 注意事项

1. **内存占用**: 计算 return 时会缓存所有轨迹的 return 值，大数据集时注意内存使用
2. **Bin 范围**: 如果 return 值经常超出范围，考虑扩大 bin_min/bin_max
3. **分布式训练**: 确保 NCCL 环境变量正确配置（RDMA、网络接口等）
4. **数据格式**: 确保数据集包含 `episode_success` 列（如果不存在，默认全部成功）

## 🐛 常见问题

**Q: 训练时出现 "trying to initialize the default process group twice!" 错误？**

A: 使用 `torchrun` 时，PyTorch 会自动初始化进程组，代码中不应再次初始化。已修复，检查 `dist.is_initialized()` 避免重复初始化。

**Q: Return 值超出 bin 范围怎么办？**

A: 代码会自动裁剪到有效范围，并记录警告。如果频繁出现，考虑扩大 `bin_min`/`bin_max`。

**Q: 如何调整训练速度？**

A: 可以调整 `--batch_size`、`--num_workers`，或使用更多 GPU。

**Q: 如何在不同数据集上使用？**

A: 修改 `run_value.sh` 中的 `DATA_MIX` 参数，或添加新的 mixture 定义到 `DATASET_NAMED_MIXTURES`。

---

**最后更新**: 2025-02-10
