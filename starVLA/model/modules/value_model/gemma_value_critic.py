from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoConfig,
    AutoTokenizer,
    Gemma3ForCausalLM,
    GemmaForCausalLM,
    SiglipVisionModel,
)
from transformers.cache_utils import DynamicCache
from transformers.models.auto import CONFIG_MAPPING

try:
    from transformers import Siglip2VisionModel
except ImportError:  # Older transformers versions only have the SigLIP v1 class.
    Siglip2VisionModel = None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_dtype(name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    normalized = str(name or "bf16").lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "16"}:
        return torch.float16
    return torch.float32


def _from_pretrained_with_dtype(model_cls, model_id: str, dtype: torch.dtype, **kwargs):
    try:
        return model_cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def _vision_model_class(model_id: str):
    try:
        model_type = str(AutoConfig.from_pretrained(model_id).model_type).lower()
    except Exception:
        model_type = ""
    if model_type in {"siglip2", "siglip2_vision_model"} and Siglip2VisionModel is not None:
        return Siglip2VisionModel
    return SiglipVisionModel


@dataclass(frozen=True)
class GemmaExpertConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_expert_config(variant: str) -> GemmaExpertConfig:
    if variant == "gemma_1m":
        return GemmaExpertConfig(
            width=128,
            depth=4,
            mlp_dim=448,
            num_heads=1,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_50m":
        return GemmaExpertConfig(
            width=384,
            depth=18,
            mlp_dim=1536,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_100m":
        return GemmaExpertConfig(
            width=512,
            depth=18,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_150m":
        return GemmaExpertConfig(
            width=640,
            depth=18,
            mlp_dim=2560,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_300m":
        return GemmaExpertConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return GemmaExpertConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown Gemma expert variant: {variant}")


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        image = image.numpy()

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.size and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(arr).convert("RGB")

    raise TypeError(f"Unsupported image type for GemmaValueCritic: {type(image)!r}")


class CategoricalValueHead(nn.Module):
    def __init__(self, hidden_size: int, num_bins: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_bins),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_size, num_bins),
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class GemmaValueCritic(nn.Module):
    """A lightweight SigLIP2 + Gemma3 value critic with a categorical value head.

    The public interface intentionally matches QwenValue:
      - forward(examples) returns {"value_loss": loss}
      - predict_value(examples, bin_min, bin_max) returns logits/probs/value
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        framework_cfg = _cfg_get(config, "framework", None)
        gemma_cfg = _cfg_get(framework_cfg, "gemma_value", {})

        self.num_bins = int(_cfg_get(framework_cfg, "value_num_bins", 201))
        self.max_text_length = int(_cfg_get(gemma_cfg, "max_text_length", 200))
        self.text_template = str(_cfg_get(gemma_cfg, "text_template", "Task: {text}."))
        self.max_images = _cfg_get(gemma_cfg, "max_images", None)
        self.max_images = None if self.max_images is None else int(self.max_images)
        self.dtype = _as_dtype(_cfg_get(gemma_cfg, "precision", "bf16"))
        self.target_loss = str(_cfg_get(gemma_cfg, "target_loss", "soft")).lower()
        if self.target_loss not in {"soft", "hard"}:
            raise ValueError(f"Unsupported gemma_value.target_loss: {self.target_loss}")
        self.bin_min = float(_cfg_get(gemma_cfg, "bin_min", -1.0))
        self.bin_max = float(_cfg_get(gemma_cfg, "bin_max", 0.0))
        if self.bin_max <= self.bin_min:
            raise ValueError(
                f"gemma_value.bin_max ({self.bin_max}) must be > bin_min ({self.bin_min})"
            )
        self.readout_type = str(_cfg_get(gemma_cfg, "readout_type", "expert")).lower()
        if self.readout_type not in {"expert", "direct"}:
            raise ValueError(f"Unsupported gemma_value.readout_type: {self.readout_type}")
        self.stop_gradient_to_vlm = bool(_cfg_get(gemma_cfg, "stop_gradient_to_vlm", False))
        self.freeze_vision_encoder = bool(
            _cfg_get(gemma_cfg, "freeze_vision_encoder", False)
        ) or self.stop_gradient_to_vlm
        self.freeze_text_model = bool(
            _cfg_get(gemma_cfg, "freeze_text_model", False)
        ) or self.stop_gradient_to_vlm
        self.freeze_image_projection = bool(
            _cfg_get(gemma_cfg, "freeze_image_projection", False)
        ) or self.stop_gradient_to_vlm

        vision_model = _cfg_get(gemma_cfg, "vision_model", "google/siglip2-so400m-patch14-224")
        text_model = _cfg_get(gemma_cfg, "text_model", "google/gemma-3-270m")
        tokenizer_model = _cfg_get(gemma_cfg, "tokenizer", text_model)

        trust_remote_code = bool(_cfg_get(gemma_cfg, "trust_remote_code", False))
        self.image_processor = AutoImageProcessor.from_pretrained(
            vision_model,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vision_tower = _from_pretrained_with_dtype(
            _vision_model_class(vision_model),
            vision_model,
            self.dtype,
            trust_remote_code=trust_remote_code,
        )
        if hasattr(self.vision_tower, "head"):
            for param in self.vision_tower.head.parameters():
                param.requires_grad = False
        self.text_model = _from_pretrained_with_dtype(
            Gemma3ForCausalLM,
            text_model,
            self.dtype,
            trust_remote_code=trust_remote_code,
        )

        if hasattr(self.text_model, "lm_head") and (
            self.text_model.lm_head.weight is not self.text_model.model.embed_tokens.weight
        ):
            for param in self.text_model.lm_head.parameters():
                param.requires_grad = False

        vision_hidden = int(self.vision_tower.config.hidden_size)
        text_hidden = int(self.text_model.config.hidden_size)
        text_head_dim = int(getattr(self.text_model.config, "head_dim", 256))
        self.image_projection = nn.Linear(
            vision_hidden,
            text_hidden,
            bias=bool(_cfg_get(gemma_cfg, "image_projection_bias", True)),
        )
        nn.init.normal_(self.image_projection.weight, std=0.02)
        if self.image_projection.bias is not None:
            nn.init.zeros_(self.image_projection.bias)

        if self.readout_type == "expert":
            expert_variant = str(_cfg_get(gemma_cfg, "critic_expert_variant", "gemma_1m"))
            self.expert_config = get_gemma_expert_config(expert_variant)
            if self.expert_config.head_dim != text_head_dim:
                raise ValueError(
                    f"Gemma expert head_dim={self.expert_config.head_dim} must match "
                    f"Gemma3 head_dim={text_head_dim} for KV-cache readout."
                )
            expert_hf_config = CONFIG_MAPPING["gemma"](
                head_dim=self.expert_config.head_dim,
                hidden_size=self.expert_config.width,
                intermediate_size=self.expert_config.mlp_dim,
                num_attention_heads=self.expert_config.num_heads,
                num_hidden_layers=self.expert_config.depth,
                num_key_value_heads=self.expert_config.num_kv_heads,
                vocab_size=int(_cfg_get(gemma_cfg, "expert_vocab_size", 257152)),
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
            )
            self.value_expert = GemmaForCausalLM(config=expert_hf_config)
            self.value_expert.model.embed_tokens = None
            if hasattr(self.value_expert, "lm_head"):
                for param in self.value_expert.lm_head.parameters():
                    param.requires_grad = False
            self.value_token = nn.Parameter(torch.zeros(1, 1, self.expert_config.width))
            value_hidden_size = self.expert_config.width
        else:
            self.expert_config = None
            self.value_expert = None
            self.value_token = nn.Parameter(torch.zeros(1, 1, text_hidden))
            value_hidden_size = text_hidden
        nn.init.normal_(self.value_token, std=0.02)

        self.value_head = CategoricalValueHead(
            hidden_size=value_hidden_size,
            num_bins=self.num_bins,
            hidden_dim=int(_cfg_get(gemma_cfg, "value_head_hidden_dim", 0)),
            dropout=float(_cfg_get(gemma_cfg, "value_dropout", 0.0)),
        )

        if self.freeze_vision_encoder:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
        if self.freeze_image_projection:
            for param in self.image_projection.parameters():
                param.requires_grad = False
        if self.freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
            if hasattr(self.text_model, "lm_head"):
                for param in self.text_model.lm_head.parameters():
                    param.requires_grad = False

        if bool(_cfg_get(gemma_cfg, "use_gradient_checkpointing", False)):
            self._enable_gradient_checkpointing()
        if self.dtype != torch.float32:
            self.to(dtype=self.dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _enable_gradient_checkpointing(self) -> None:
        if hasattr(self.vision_tower, "gradient_checkpointing_enable"):
            self.vision_tower.gradient_checkpointing_enable()
        if hasattr(self.text_model, "gradient_checkpointing_enable"):
            self.text_model.gradient_checkpointing_enable()
        if hasattr(self.text_model, "config"):
            self.text_model.config.use_cache = False

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :].to(device=self.device, dtype=torch.bool)
        dtype = next(self.text_model.parameters()).dtype
        min_value = torch.finfo(dtype).min
        return torch.where(
            att_2d_masks_4d,
            torch.tensor(0.0, dtype=dtype, device=self.device),
            torch.tensor(min_value, dtype=dtype, device=self.device),
        )

    @staticmethod
    def _detach_dynamic_cache(cache):
        if cache is None:
            return cache
        if isinstance(cache, DynamicCache) and hasattr(cache, "key_cache"):
            for idx in range(len(cache.key_cache)):
                cache.key_cache[idx] = cache.key_cache[idx].detach()
                cache.value_cache[idx] = cache.value_cache[idx].detach()
        return cache

    def _normalize_images(self, image_field: Any) -> list[Image.Image]:
        images = image_field if isinstance(image_field, (list, tuple)) else [image_field]
        if self.max_images is not None:
            images = images[: self.max_images]
        if not images:
            raise ValueError("GemmaValueCritic expects at least one image per example.")
        return [_to_pil_image(img) for img in images]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_tower.eval()
        if self.freeze_text_model:
            self.text_model.eval()
        return self

    def _encode_images(self, examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        image_groups = [self._normalize_images(ex["image"]) for ex in examples]
        counts = [len(group) for group in image_groups]
        flat_images = [img for group in image_groups for img in group]

        vision_inputs = self.image_processor(images=flat_images, return_tensors="pt")
        vision_inputs = {
            key: value.to(
                device=self.device,
                dtype=self.dtype if torch.is_floating_point(value) else value.dtype,
            )
            for key, value in vision_inputs.items()
        }

        vision_out = self.vision_tower(**vision_inputs, return_dict=True)
        image_tokens = self.image_projection(vision_out.last_hidden_state.to(self.image_projection.weight.dtype))

        split_tokens = torch.split(image_tokens, counts, dim=0)
        flattened = [tokens.reshape(-1, tokens.shape[-1]) for tokens in split_tokens]
        max_len = max(tokens.shape[0] for tokens in flattened)
        hidden_size = flattened[0].shape[-1]

        padded = image_tokens.new_zeros((len(examples), max_len, hidden_size))
        mask = torch.zeros((len(examples), max_len), dtype=torch.long, device=self.device)
        for idx, tokens in enumerate(flattened):
            length = tokens.shape[0]
            padded[idx, :length] = tokens
            mask[idx, :length] = 1
        return padded, mask

    def _encode_text(self, examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [self._format_text(ex) for ex in examples]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        text_embeds = self.text_model.model.embed_tokens(input_ids)
        return text_embeds, attention_mask

    def _format_text(self, example: dict) -> str:
        text = str(example.get("lang", example.get("language", example.get("prompt", ""))))
        text = text.lower().strip().replace("_", " ").replace("\n", " ")
        if text and text[-1] in string.punctuation and text[-1] not in "\"'":
            text = text[:-1]
        if self.text_template:
            return self.text_template.format(text=text)
        return text

    def _compute_logits(self, examples: list[dict]) -> torch.Tensor:
        if not isinstance(examples, list):
            examples = [examples]

        image_embeds, image_mask = self._encode_images(examples)
        text_embeds, text_mask = self._encode_text(examples)
        if self.readout_type == "expert":
            return self._compute_logits_expert(image_embeds, image_mask, text_embeds, text_mask)
        return self._compute_logits_direct(image_embeds, image_mask, text_embeds, text_mask)

    def _compute_logits_direct(
        self,
        image_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = text_embeds.shape[0]
        value_token = self.value_token.to(dtype=text_embeds.dtype).expand(batch_size, -1, -1)
        value_mask = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)

        inputs_embeds = torch.cat(
            [
                image_embeds.to(dtype=text_embeds.dtype),
                text_embeds,
                value_token,
            ],
            dim=1,
        )
        attention_mask = torch.cat([image_mask, text_mask, value_mask], dim=1)

        outputs = self.text_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        value_hidden = outputs.last_hidden_state[:, -1, :]
        return self.value_head(value_hidden.to(dtype=self.value_head.net[-1].weight.dtype))

    def _compute_logits_expert(
        self,
        image_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.value_expert is None:
            raise RuntimeError("GemmaValueCritic readout_type='expert' requires value_expert.")

        prefix_embs = torch.cat(
            [image_embeds.to(dtype=text_embeds.dtype), text_embeds],
            dim=1,
        )
        prefix_mask = torch.cat([image_mask, text_mask], dim=1).to(dtype=torch.long)

        prefix_attn = prefix_mask[:, None, :].bool() & prefix_mask[:, :, None].bool()
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = (torch.cumsum(prefix_mask, dim=1) - 1).clamp_min(0)

        prefix_out = self.text_model.model(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
        past_kv = prefix_out.past_key_values
        if self.stop_gradient_to_vlm:
            past_kv = self._detach_dynamic_cache(past_kv)

        batch_size = prefix_mask.shape[0]
        suffix_embs = self.value_token.to(
            device=self.device,
            dtype=next(self.value_expert.parameters()).dtype,
        ).expand(batch_size, -1, -1)

        prefix_to_suffix = prefix_mask[:, None, :].bool()
        suffix_self = torch.ones((batch_size, 1, 1), dtype=torch.bool, device=self.device)
        suffix_attn = torch.cat([prefix_to_suffix, suffix_self], dim=2)
        suffix_attn_4d = self._prepare_attention_masks_4d(suffix_attn)
        suffix_pos = prefix_mask.sum(dim=1, keepdim=True)

        suffix_out = self.value_expert.model(
            inputs_embeds=suffix_embs,
            attention_mask=suffix_attn_4d,
            position_ids=suffix_pos,
            past_key_values=past_kv,
            use_cache=False,
            return_dict=True,
        )
        value_hidden = suffix_out.last_hidden_state[:, -1, :]
        logits = self.value_head(value_hidden.to(dtype=self.value_head.net[-1].weight.dtype))
        if self.training and not self.stop_gradient_to_vlm and prefix_out.last_hidden_state.requires_grad:
            logits = logits + prefix_out.last_hidden_state.sum() * 0.0
        return logits

    def forward(self, examples: list[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        logits = self._compute_logits(examples)
        if self.target_loss == "soft" and "value_target" in examples[0]:
            targets = torch.tensor(
                np.array([ex["value_target"] for ex in examples]),
                device=logits.device,
                dtype=torch.float32,
            )
            value_loss = self._soft_categorical_loss(logits, targets)
        elif "value_bin" in examples[0]:
            targets = torch.tensor(
                np.array([ex["value_bin"] for ex in examples]),
                device=logits.device,
                dtype=torch.long,
            )
            value_loss = F.cross_entropy(logits.float(), targets)
        else:
            value_loss = logits.mean() * 0.0
        return {"value_loss": value_loss}

    def _soft_categorical_loss(self, logits: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        target_values = target_values.float().view(-1).clamp(self.bin_min, self.bin_max)
        delta = (self.bin_max - self.bin_min) / max(self.num_bins - 1, 1)
        b = (target_values - self.bin_min) / delta
        lower = b.floor().long().clamp(0, self.num_bins - 1)
        upper = b.ceil().long().clamp(0, self.num_bins - 1)
        d_to_lower = b - lower.float()
        d_to_upper = upper.float() - b
        same_bin = lower == upper
        d_to_lower = torch.where(same_bin, torch.zeros_like(d_to_lower), d_to_lower)
        d_to_upper = torch.where(same_bin, torch.ones_like(d_to_upper), d_to_upper)

        target_probs = torch.zeros(
            target_values.shape[0],
            self.num_bins,
            dtype=logits.dtype,
            device=logits.device,
        )
        batch_idx = torch.arange(target_values.shape[0], device=logits.device)
        target_probs[batch_idx, lower] += d_to_upper.to(target_probs.dtype)
        target_probs[batch_idx, upper] += d_to_lower.to(target_probs.dtype)
        return -(target_probs.float() * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()

    @torch.inference_mode()
    def predict_value(
        self,
        examples: list[dict],
        bin_min: Optional[float] = None,
        bin_max: Optional[float] = None,
        **kwargs,
    ) -> dict:
        logits = self._compute_logits(examples)
        probs = torch.softmax(logits.float(), dim=-1)
        bin_index = torch.argmax(probs, dim=-1)
        result = {
            "logits": logits.detach().float().cpu().numpy(),
            "probs": probs.detach().cpu().numpy(),
            "bin_index": bin_index.detach().cpu().numpy(),
        }
        out_bin_min = self.bin_min if bin_min is None else float(bin_min)
        out_bin_max = self.bin_max if bin_max is None else float(bin_max)
        if out_bin_min is not None and out_bin_max is not None:
            bin_delta = (out_bin_max - out_bin_min) / max(self.num_bins - 1, 1)
            bin_values = (
                torch.arange(self.num_bins, dtype=torch.float32, device=probs.device)
                * bin_delta
                + out_bin_min
            )
            values = torch.sum(probs * bin_values.unsqueeze(0), dim=-1)
            result["values"] = values.detach().cpu().numpy()
        return result
