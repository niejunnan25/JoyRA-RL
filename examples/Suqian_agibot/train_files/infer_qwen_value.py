"""
QwenValue 模型推理脚本

用法:
    python infer_qwen_value.py \
        --checkpoint_path outputs_value_aloha_agilex_550_mix/qwen_value_best.pt \
        --config_yaml examples/Suqian_agibot/train_files/starvla_value_function.yaml \
        --image_path path/to/image.jpg \
        --instruction "Your task instruction here" \
        --bin_min -600.0 \
        --bin_max 0.0
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework


def parse_args():
    parser = argparse.ArgumentParser(description="QwenValue 模型推理")
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="训练好的 checkpoint 路径（例如: outputs_value_aloha_agilex_550_mix/qwen_value_best.pt）",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="配置文件路径（与训练时使用的相同）",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="输入图像路径。可以是：1) 单个图像文件；2) 图像目录（会自动查找所有图像）；3) 逗号分隔的多个图像路径（按顺序：高视角,左手腕,右手腕）",
    )
    parser.add_argument(
        "--image_high",
        type=str,
        default=None,
        help="高视角图像路径（如果单独指定）",
    )
    parser.add_argument(
        "--image_left_wrist",
        type=str,
        default=None,
        help="左手腕视角图像路径（如果单独指定）",
    )
    parser.add_argument(
        "--image_right_wrist",
        type=str,
        default=None,
        help="右手腕视角图像路径（如果单独指定）",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="任务指令文本",
    )
    parser.add_argument(
        "--bin_min",
        type=float,
        default=-600.0,
        help="Value bin 的最小值（与训练时一致）",
    )
    parser.add_argument(
        "--bin_max",
        type=float,
        default=0.0,
        help="Value bin 的最大值（与训练时一致）",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=201,
        help="Value bin 的数量（与训练时一致，默认201）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备（cuda 或 cpu）",
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """加载训练好的模型"""
    print(f"加载配置文件: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # 确保使用 QwenValue
    cfg.framework.name = "QwenValue"
    
    print(f"构建模型...")
    model = build_framework(cfg)
    
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理 checkpoint 格式
    if "model_state_dict" in checkpoint:
        # 训练时保存的完整 checkpoint
        state_dict = checkpoint["model_state_dict"]
        print(f"  - Step: {checkpoint.get('step', 'N/A')}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Loss: {checkpoint.get('loss', 'N/A')}")
    else:
        # 直接保存的 state_dict
        state_dict = checkpoint
    
    # 移除 DDP 前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"模型已加载到 {device}")
    return model


def load_images(image_path: str, image_high: str = None, image_left_wrist: str = None, image_right_wrist: str = None):
    """
    加载多视角图像
    
    返回: List[PIL.Image]，顺序为 [高视角, 左手腕视角, 右手腕视角]
    """
    images = []
    
    # 方式1: 如果单独指定了三个视角
    if image_high and image_left_wrist and image_right_wrist:
        print(f"使用单独指定的三个视角:")
        print(f"  高视角: {image_high}")
        print(f"  左手腕: {image_left_wrist}")
        print(f"  右手腕: {image_right_wrist}")
        images.append(Image.open(image_high).convert("RGB"))
        images.append(Image.open(image_left_wrist).convert("RGB"))
        images.append(Image.open(image_right_wrist).convert("RGB"))
        return images
    
    # 方式2: 如果 image_path 是逗号分隔的多个路径
    if "," in image_path:
        paths = [p.strip() for p in image_path.split(",")]
        if len(paths) == 3:
            print(f"使用逗号分隔的三个图像路径:")
            for i, path in enumerate(paths):
                print(f"  视角 {i+1}: {path}")
            for path in paths:
                images.append(Image.open(path).convert("RGB"))
            return images
        else:
            raise ValueError(f"逗号分隔的图像路径必须是3个，当前有 {len(paths)} 个")
    
    # 方式3: 如果是目录，查找所有图像文件
    image_path_obj = Path(image_path)
    if image_path_obj.is_dir():
        image_files = sorted(list(image_path_obj.glob("*.jpg")) + list(image_path_obj.glob("*.png")))
        if len(image_files) >= 3:
            print(f"从目录 {image_path} 中找到 {len(image_files)} 个图像，使用前3个:")
            for i, img_file in enumerate(image_files[:3]):
                print(f"  视角 {i+1}: {img_file.name}")
            for img_file in image_files[:3]:
                images.append(Image.open(img_file).convert("RGB"))
            return images
        elif len(image_files) == 1:
            print(f"警告: 目录中只有1个图像，将使用单个图像（训练时使用3个视角）")
            return [Image.open(image_files[0]).convert("RGB")]
        else:
            raise ValueError(f"目录 {image_path} 中需要至少3个图像文件，当前只有 {len(image_files)} 个")
    
    # 方式4: 单个图像文件（兼容旧版本，但会警告）
    elif image_path_obj.is_file():
        print(f"警告: 只提供了单个图像文件。训练时使用3个视角（高视角、左手腕、右手腕），")
        print(f"      建议使用 --image_high, --image_left_wrist, --image_right_wrist 或逗号分隔的路径")
        return [Image.open(image_path_obj).convert("RGB")]
    
    else:
        raise ValueError(f"无效的图像路径: {image_path}")


def predict_value(
    model,
    image_path: str,
    instruction: str,
    bin_min: float,
    bin_max: float,
    num_bins: int,
    device: str = "cuda",
    image_high: str = None,
    image_left_wrist: str = None,
    image_right_wrist: str = None,
):
    """使用模型预测 value"""
    # 加载多视角图像
    images = load_images(image_path, image_high, image_left_wrist, image_right_wrist)
    
    # 构造输入（image 应该是列表）
    example = {
        "image": images,  # List[PIL.Image]，顺序为 [高视角, 左手腕, 右手腕]
        "lang": instruction,
    }
    
    print(f"\n输入:")
    print(f"  - 图像数量: {len(images)} 个视角")
    if len(images) == 3:
        print(f"    * 视角1 (高视角): {images[0].size}")
        print(f"    * 视角2 (左手腕): {images[1].size}")
        print(f"    * 视角3 (右手腕): {images[2].size}")
    else:
        print(f"    * 警告: 训练时使用3个视角，当前只有 {len(images)} 个")
    print(f"  - 指令: {instruction}")
    
    # 推理
    with torch.no_grad():
        result = model.predict_value(
            examples=[example],
            bin_min=bin_min,
            bin_max=bin_max,
        )
    
    # 解析结果
    values = result["values"]  # [B]
    probs = result["probs"]  # [B, num_bins]
    bin_index = result["bin_index"]  # [B]
    
    value = values[0]  # 单个样本
    bin_probs = probs[0]  # [num_bins]
    bin_idx = bin_index[0]
    
    print(f"\n预测结果:")
    print(f"  - Value: {value:.4f}")
    print(f"  - Bin Index: {bin_idx} / {num_bins-1}")
    print(f"  - Bin Probability: {bin_probs[bin_idx]:.4f}")
    print(f"  - Value Range: [{bin_min:.2f}, {bin_max:.2f}]")
    
    # 显示 top-5 bins
    top5_indices = bin_probs.argsort()[-5:][::-1]
    print(f"\nTop-5 Value Bins:")
    for i, idx in enumerate(top5_indices):
        bin_value = bin_min + (bin_max - bin_min) * idx / (num_bins - 1)
        print(f"  {i+1}. Bin {idx}: value={bin_value:.4f}, prob={bin_probs[idx]:.4f}")
    
    return {
        "value": value,
        "bin_idx": bin_idx,
        "bin_probs": bin_probs,
    }


def main():
    args = parse_args()
    
    # 加载模型
    model = load_model(args.checkpoint_path, args.config_yaml, args.device)
    
    # 预测
    result = predict_value(
        model=model,
        image_path=args.image_path,
        instruction=args.instruction,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        num_bins=args.num_bins,
        device=args.device,
        image_high=args.image_high,
        image_left_wrist=args.image_left_wrist,
        image_right_wrist=args.image_right_wrist,
    )
    
    print("\n推理完成！")


if __name__ == "__main__":
    main()
