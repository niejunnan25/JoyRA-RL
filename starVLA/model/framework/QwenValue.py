import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保 workspace 根目录在 sys.path 里（与原 QwenPerceiver 一致）
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch


logger = initialize_overwatch(__name__)


class ValueHead(nn.Module):
    """
    输入 Qwen-VL 的 last_hidden [B, L, H]，做 pooling 后输出 201-bin 的 logits。

    即：不再直接回归标量 V(o)，而是输出一个长度为 num_bins 的分类分布，
    对应论文中将 return 离散为 B=201 个 bin，再用交叉熵训练的做法。
    """

    def __init__(self, hidden_dim: int, num_bins: int = 201):
        super().__init__()
        self.num_bins = num_bins
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden: [B, L, H]
        Returns:
            logits: [B, num_bins] 每个样本对应的 value-bin logits
        """
        # mean pooling，可以后续替换为 CLS / attention pool
        pooled = last_hidden.mean(dim=1)  # [B, H]
        logits = self.mlp(pooled)  # [B, num_bins]
        return logits


@FRAMEWORK_REGISTRY.register("QwenValue")
class Qwen_Value(baseframework):
    """
    QwenValue
    - Qwen-VL 编码器（get_vlm_model），顶层接一个 ValueHead 输出 201-bin logits
    - forward 中期望 dataloader 提供 "value_bin"（整数 bin 标签），使用交叉熵训练
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        hidden_dim = self.qwen_vl_interface.model.config.hidden_size

        # 这里显式冻结 Qwen3-VL 的 lm_head，使其不再是可训练参数（不会被 DDP 跟踪）。
        base_vlm = getattr(self.qwen_vl_interface, "model", None)
        if base_vlm is not None and hasattr(base_vlm, "lm_head"):
            for p in base_vlm.lm_head.parameters():
                p.requires_grad = False

        # bin 个数可从 config.framework.value_num_bins 覆盖，默认 201
        num_bins = 201
        if self.config is not None and hasattr(self.config, "framework"):
            try:
                num_bins = int(getattr(self.config.framework, "value_num_bins", 201))
            except Exception:
                num_bins = 201
        self.num_bins = num_bins

        self.value_head = ValueHead(hidden_dim=hidden_dim, num_bins=self.num_bins)

    def _encode_observations(
        self,
        examples: List[dict],
    ) -> torch.Tensor:
        """
        将 batch 中的 image / lang 编码成 Qwen-VL 的 last_hidden。

        Args:
            examples: List[dict]，每个 dict 至少包含:
                - "image": 图像或图像列表
                - "lang": 指令字符串

        Returns:
            last_hidden: [B, L, H]
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H], bfloat16

        # 将 encoder 输出转换为 float32，避免后续 Linear 权重 (float32) 与输入 (bfloat16) dtype 不匹配
        last_hidden = last_hidden.to(dtype=torch.float32)

        return last_hidden

    # ===== 训练前向 =====
    def forward(
        self,
        examples: List[dict],
        **kwargs,
    ) -> dict:
        """
        训练前向：
          - dataloader 需要在每个 example 中提供 "value_bin"（int，0 ~ num_bins-1），
            即离散化后的 return bin 标签。

        Returns:
            dict:
                - "value_loss": 标量 loss（交叉熵），用于优化
        """
        if not isinstance(examples, list):
            examples = [examples]

        last_hidden = self._encode_observations(examples)  # [B, L, H]

        logits = self.value_head(last_hidden)  # [B, num_bins]

        if "value_bin" not in examples[0]:
            logger.warning(
                "[QwenValue] 'value_bin' not found in examples; returning zero loss."
            )
            value_loss = logits.mean() * 0.0
        else:
            targets = torch.tensor(
                np.array([ex["value_bin"] for ex in examples]),
                device=logits.device,
                dtype=torch.long,
            )  # [B]
            value_loss = F.cross_entropy(logits, targets)

        return {"value_loss": value_loss}

    @torch.inference_mode()
    def predict_value(
        self,
        examples: List[dict],
        bin_min: Optional[float] = None,
        bin_max: Optional[float] = None,
        **kwargs,
    ) -> dict:
        """
        推理接口：给定一批样本，返回 value-bin 的分布与连续 value。

        按照 π₀.₆\* 论文的方式，从学到的分布中提取连续 value：
            V^π_ref(o_t, ℓ) = Σ_{b∈[0,B]} p_φ(V=b|o_t) v(b)
        其中 v(b) 是 bin b 对应的 value。

        Args:
            examples: 与 forward 相同的输入格式（无需包含 value_target）。
            bin_min: bin 的最小值（用于将 bin_index 转换为连续 value）。如果为 None，只返回 bin_index。
            bin_max: bin 的最大值（用于将 bin_index 转换为连续 value）。如果为 None，只返回 bin_index。

        Returns:
            dict:
                - "logits": np.ndarray, shape [B, num_bins]，每个 bin 的原始 logits
                - "probs": np.ndarray, shape [B, num_bins]，每个 bin 的概率分布
                - "bin_index": np.ndarray[int64], shape [B]，每个样本的 argmax bin（0 ~ num_bins-1）
                - "values": np.ndarray[float32], shape [B]，连续 value（如果提供了 bin_min/bin_max）
        """
        if not isinstance(examples, list):
            examples = [examples]

        last_hidden = self._encode_observations(examples)  # [B, L, H]
        with torch.autocast("cuda", dtype=torch.float32):
            logits = self.value_head(last_hidden)  # [B, num_bins]

        probs = torch.softmax(logits, dim=-1)  # [B, num_bins]
        bin_index = torch.argmax(probs, dim=-1)  # [B]

        result = {
            "logits": logits.detach().cpu().numpy(),
            "probs": probs.detach().cpu().numpy(),
            "bin_index": bin_index.detach().cpu().numpy(),
        }

        # 如果提供了 bin 范围，计算连续 value（按 π₀.₆\* 论文的期望值方式）
        if bin_min is not None and bin_max is not None:
            # 计算每个 bin 对应的 value: v(b) = bin_min + b * bin_delta
            bin_delta = (bin_max - bin_min) / max(self.num_bins - 1, 1)
            bin_values = torch.arange(
                0, self.num_bins, dtype=torch.float32, device=probs.device
            ) * bin_delta + bin_min  # [num_bins]

            # 按论文公式：V = Σ_b p(b) * v(b)
            values = torch.sum(probs * bin_values.unsqueeze(0), dim=-1)  # [B]

            result["values"] = values.detach().cpu().numpy()

        return result
