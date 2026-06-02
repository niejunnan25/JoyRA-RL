"""
绘制训练 loss 曲线图

用法:
    python plot_training_losses.py --loss_file outputs_value_qwen/training_losses.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_losses(loss_file):
    """加载 loss 记录"""
    with open(loss_file, "r") as f:
        data = json.load(f)
    return data.get("train_losses", []), data.get("val_losses", [])


def plot_losses(train_losses, val_losses, output_path=None, log_y_train: bool = False):
    """绘制 loss 曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 左图：训练 loss（按 step）
    if train_losses:
        train_steps = [x["step"] for x in train_losses]
        train_loss_values = [x["loss"] for x in train_losses]
        
        axes[0].plot(train_steps, train_loss_values, alpha=0.6, linewidth=1, label="Train Loss")
        
        # 计算移动平均（窗口大小=100）
        if len(train_loss_values) > 100:
            window = 100
            moving_avg = np.convolve(train_loss_values, np.ones(window)/window, mode='valid')
            moving_steps = train_steps[window-1:]
            axes[0].plot(moving_steps, moving_avg, linewidth=2, label=f"Moving Avg (window={window})", color='red')
        
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        if log_y_train:
            # log 坐标要求数据为正；若存在 <=0 的 loss，matplotlib 会报错
            if np.min(train_loss_values) <= 0:
                print("警告: train loss 中存在 <=0 值，无法使用对数坐标，已跳过 log y。")
            else:
                axes[0].set_yscale("log")
    
    # 右图：训练和验证 loss（按 epoch）
    if train_losses:
        # 按 epoch 分组计算平均训练 loss
        epoch_train_losses = {}
        for x in train_losses:
            epoch = x["epoch"]
            if epoch not in epoch_train_losses:
                epoch_train_losses[epoch] = []
            epoch_train_losses[epoch].append(x["loss"])
        
        epochs = sorted(epoch_train_losses.keys())
        avg_train_losses = [np.mean(epoch_train_losses[e]) for e in epochs]
        
        axes[1].plot(epochs, avg_train_losses, marker='o', label="Train Loss (avg per epoch)", linewidth=2)
    
    if val_losses:
        val_epochs = [x["epoch"] for x in val_losses]
        val_loss_values = [x["loss"] for x in val_losses]
        axes[1].plot(val_epochs, val_loss_values, marker='s', label="Val Loss", linewidth=2, color='orange')
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training & Validation Loss (per Epoch)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="绘制训练 loss 曲线")
    parser.add_argument(
        "--loss_file",
        type=str,
        required=True,
        help="loss 记录 JSON 文件路径（例如: outputs_value_qwen/training_losses.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径（例如: loss_curve.png）。如果不指定，将显示图表。",
    )
    parser.add_argument(
        "--log_y_train",
        action="store_true",
        help="若启用，左图（按 step 的训练 loss）使用对数 y 轴。",
    )
    
    args = parser.parse_args()
    
    loss_file = Path(args.loss_file)
    if not loss_file.exists():
        print(f"错误: 文件不存在: {loss_file}")
        return
    
    print(f"加载 loss 记录: {loss_file}")
    train_losses, val_losses = load_losses(loss_file)
    
    print(f"  - 训练 loss 记录数: {len(train_losses)}")
    print(f"  - 验证 loss 记录数: {len(val_losses)}")
    
    if not train_losses and not val_losses:
        print("警告: 没有找到任何 loss 记录")
        return
    
    output_path = Path(args.output) if args.output else None
    plot_losses(train_losses, val_losses, output_path, log_y_train=args.log_y_train)


if __name__ == "__main__":
    main()
