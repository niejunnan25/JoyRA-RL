from __future__ import annotations

from typing import Optional

import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.value_model import GemmaValueCritic
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("GemmaValue")
class Gemma_Value(baseframework):
    """SigLIP2 + Gemma3 value function framework.

    This class mirrors the public QwenValue API so train/eval/visualization
    code can switch critic backends via `framework.name`.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.value_model = GemmaValueCritic(config=config)
        self.num_bins = self.value_model.num_bins

    def forward(self, examples: list[dict], **kwargs) -> dict:
        return self.value_model(examples, **kwargs)

    @torch.inference_mode()
    def predict_value(self, examples: list[dict], **kwargs) -> dict:
        return self.value_model.predict_value(examples, **kwargs)
