# QwenValue 模型推理使用说明

## 快速开始

### 1. 基本用法

```bash
cd /mnt/workspace/users/daiyixiang/JoyRA-RL

python examples/Suqian_agibot/train_files/infer_qwen_value.py \
    --checkpoint_path outputs_value_aloha_agilex_550_mix/qwen_value_best.pt \
    --config_yaml examples/Suqian_agibot/train_files/starvla_value_function.yaml \
    --image_path /path/to/your/image.jpg \
    --instruction "Your task instruction here" \
    --bin_min -600.0 \
    --bin_max 0.0
```

### 2. 参数说明

- `--checkpoint_path`: 训练好的模型 checkpoint 路径
  - 可以使用 `qwen_value_best.pt`（验证集最佳模型）
  - 或 `qwen_value_final.pt`（最终模型）
  - 或 `checkpoint_step_XXXXX.pt`（按步数保存的 checkpoint）

- `--config_yaml`: 训练时使用的配置文件路径

- `--image_path`: 输入图像路径（支持多种方式）
  - **方式1**: 逗号分隔的3个图像路径（按顺序：高视角,左手腕,右手腕）
    - 例如: `--image_path "high.jpg,left_wrist.jpg,right_wrist.jpg"`
  - **方式2**: 图像目录（会自动使用目录中前3个图像文件，按文件名排序）
  - **方式3**: 单个图像文件（会警告，因为训练时使用3个视角）

- `--image_high`: 高视角图像路径（可选，如果单独指定）
- `--image_left_wrist`: 左手腕视角图像路径（可选，如果单独指定）
- `--image_right_wrist`: 右手腕视角图像路径（可选，如果单独指定）

**注意**: 训练时使用3个视角（高视角、左手腕、右手腕），推理时也应该提供3个视角以获得最佳效果。

- `--instruction`: 任务指令文本（与训练时的格式一致）

- `--bin_min`: Value bin 的最小值（与训练时一致，默认 -600.0）

- `--bin_max`: Value bin 的最大值（与训练时一致，默认 0.0）

- `--num_bins`: Value bin 的数量（与训练时一致，默认 201）

- `--device`: 推理设备（默认 cuda，如果不可用则使用 cpu）

### 3. 输出说明

脚本会输出：
- **Value**: 预测的连续 value 值（在 bin_min 到 bin_max 之间）
- **Bin Index**: 预测的 bin 索引（0 到 num_bins-1）
- **Bin Probability**: 该 bin 的概率
- **Top-5 Value Bins**: 概率最高的 5 个 bin 及其对应的 value

### 4. 示例

```bash
# 方式1: 使用逗号分隔的3个图像路径（推荐）
python examples/Suqian_agibot/train_files/infer_qwen_value.py \
    --checkpoint_path outputs_value_aloha_agilex_550_mix/qwen_value_best.pt \
    --config_yaml examples/Suqian_agibot/train_files/starvla_value_function.yaml \
    --image_path "high.jpg,left_wrist.jpg,right_wrist.jpg" \
    --instruction "pick up the bottle" \
    --bin_min -600.0 \
    --bin_max 0.0

# 方式2: 使用单独的参数指定三个视角（推荐）
python examples/Suqian_agibot/train_files/infer_qwen_value.py \
    --checkpoint_path outputs_value_aloha_agilex_550_mix/qwen_value_best.pt \
    --config_yaml examples/Suqian_agibot/train_files/starvla_value_function.yaml \
    --image_high /path/to/image_high.jpg \
    --image_left_wrist /path/to/image_left_wrist.jpg \
    --image_right_wrist /path/to/image_right_wrist.jpg \
    --instruction "pick up the bottle" \
    --bin_min -600.0 \
    --bin_max 0.0

# 方式3: 使用图像目录（会自动使用目录中前3个图像）
python examples/Suqian_agibot/train_files/infer_qwen_value.py \
    --checkpoint_path outputs_value_aloha_agilex_550_mix/qwen_value_best.pt \
    --config_yaml examples/Suqian_agibot/train_files/starvla_value_function.yaml \
    --image_path /path/to/images_directory/ \
    --instruction "pick up the bottle" \
    --bin_min -600.0 \
    --bin_max 0.0
```

### 5. 注意事项

1. **bin_min 和 bin_max 必须与训练时一致**，否则预测的 value 值会不准确
2. **图像视角**: 训练时使用3个视角（高视角、左手腕、右手腕），推理时也应该提供3个视角以获得最佳效果
3. 图像格式需要与训练时一致（通常是 RGB 格式，224x224 分辨率）
4. 指令文本格式建议与训练数据保持一致
5. 如果使用 GPU，确保有足够的显存（模型约 8-24GB）
6. 图像顺序很重要：如果使用逗号分隔或目录方式，确保图像按正确顺序（高视角、左手腕、右手腕）

### 6. 批量推理

如果需要批量推理多个图像，可以修改脚本或使用循环：

```bash
# 示例：批量推理
for img in /path/to/images/*.jpg; do
    python examples/Suqian_agibot/train_files/infer_qwen_value.py \
        --checkpoint_path outputs_value_aloha_agilex_550_mix/qwen_value_best.pt \
        --config_yaml examples/Suqian_agibot/train_files/starvla_value_function.yaml \
        --image_path "$img" \
        --instruction "your task instruction" \
        --bin_min -600.0 \
        --bin_max 0.0
done
```
