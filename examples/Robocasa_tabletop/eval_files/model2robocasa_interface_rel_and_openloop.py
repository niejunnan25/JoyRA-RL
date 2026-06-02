# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Callable, List, Tuple

import numpy as np
import cv2 as cv

# ---- project imports ----
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler
from starVLA.model.framework.share_tools import read_mode_config

# open-loop extras
import time
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import tyro
from starVLA.model.framework.base_framework import baseframework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

# ============================================================
# 0) Shared utils (ONE truth for BOTH sim-step and open-loop)
# ============================================================


def _as_np_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _inv_minmax(xn: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Inverse of normalize-to-[-1,1], with broadcasting support.
    """
    xn = np.clip(xn, -1.0, 1.0).astype(np.float32)
    lo = _as_np_f32(lo)
    hi = _as_np_f32(hi)
    denom = np.maximum(hi - lo, eps)
    return (xn + 1.0) * 0.5 * denom + lo


def normalize_with_mask_minmax_to_minus1_1(x: np.ndarray, stats: dict, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize x using stats min/max to [-1,1], with optional mask:
      - mask=True  -> normalize
      - mask=False -> keep raw value (like your action unnormalize does the opposite)

    Supports x shaped as:
      - (B,D)
      - (B,1,D)
      - (D,)
    """
    x = _as_np_f32(x)
    lo = _as_np_f32(stats["min"])
    hi = _as_np_f32(stats["max"])

    # ensure last dim alignment
    D = x.shape[-1] if x.ndim > 0 else lo.shape[0]
    lo = lo[:D]
    hi = hi[:D]

    denom = np.maximum(hi - lo, eps)

    mask = stats.get("mask", None)
    if mask is None:
        mask = np.ones((D,), dtype=bool)
    mask = np.asarray(mask, dtype=bool)[:D]

    # broadcast for x
    if x.ndim == 1:
        m = mask
    elif x.ndim == 2:
        m = mask[None, :]
    elif x.ndim == 3:
        m = mask[None, None, :]
    else:
        raise ValueError(f"Unsupported x.ndim={x.ndim}, shape={x.shape}")

    x01 = (x - lo) / denom
    xn = x01 * 2.0 - 1.0
    return np.where(m, xn, x).astype(np.float32)


def _stats_to_btd(arr: np.ndarray, T_use: int, D: int) -> np.ndarray:
    """
    Convert stats array to (1, T_use, D) for broadcasting with (B,T_use,D).
    arr can be:
      - (D,)
      - (T,D)
    """
    a = _as_np_f32(arr)
    if a.ndim == 1:
        a = a[:D][None, None, :]     # (1,1,D)
        a = np.repeat(a, T_use, axis=1)     # (1,T,D)
        return a
    if a.ndim == 2:
        a = a[:T_use, :D][None, :, :]     # (1,T,D)
        return a
    raise ValueError(f"Unsupported stats shape: {a.shape}")


def load_norm_stats(ckpt_path: str, unnorm_key: Optional[str]) -> Tuple[dict, str]:
    _, ns = read_mode_config(Path(ckpt_path))
    if unnorm_key is None:
        assert len(ns) == 1, f"Multiple datasets in norm_stats: {list(ns.keys())}, please pass unnorm_key"
        unnorm_key = next(iter(ns.keys()))
    assert unnorm_key in ns, f"unnorm_key={unnorm_key} not in norm_stats keys: {list(ns.keys())}"
    return ns[unnorm_key], unnorm_key


def load_action_stats(ckpt_path: str, unnorm_key: Optional[str]) -> dict:
    ns, _ = load_norm_stats(ckpt_path, unnorm_key)
    return ns["action"]


def load_state_stats(ckpt_path: str, unnorm_key: Optional[str]) -> dict:
    ns, _ = load_norm_stats(ckpt_path, unnorm_key)
    return ns["state"]


def load_rel_arm_stats(ckpt_path: str, unnorm_key: Optional[str]) -> dict:
    """
    Expect flat keys:
      relative_action.left_arm
      relative_action.right_arm
    """
    ns, _ = load_norm_stats(ckpt_path, unnorm_key)

    def pack(k: str) -> dict:
        obj = ns[k]
        return {"min": _as_np_f32(obj["min"]), "max": _as_np_f32(obj["max"])}

    assert "relative_action.left_arm" in ns and "relative_action.right_arm" in ns, \
        f"Missing rel stats keys. Available: {list(ns.keys())}"
    return {"left_arm": pack("relative_action.left_arm"), "right_arm": pack("relative_action.right_arm")}


def unnormalize_actions_with_mask(
    na_29: np.ndarray,     # (B,T,29) or (T,29)
    action_stats: dict,
) -> np.ndarray:
    """
    EXACT policy:
      - clip to [-1,1]
      - inv-minmax on dims where mask=True
      - keep normalized value on dims where mask=False
    """
    na = np.clip(na_29, -1.0, 1.0).astype(np.float32)
    lo = _as_np_f32(action_stats["min"])
    hi = _as_np_f32(action_stats["max"])
    assert lo.shape[0] >= 29 and hi.shape[0] >= 29, f"action_stats dim too small: {lo.shape[0]}"
    lo = lo[:29]
    hi = hi[:29]

    mask = action_stats.get("mask", None)
    if mask is None:
        mask = np.ones((29,), dtype=bool)
    mask = np.asarray(mask, dtype=bool)[:29]

    inv = _inv_minmax(na, lo, hi)
    if na.ndim == 3:
        m = mask[None, None, :]
    else:
        m = mask[None, :]
    return np.where(m, inv, na).astype(np.float32)


def postprocess_actions_rel2abs(
        normalized_actions_32: np.ndarray,     # (B,T,32)
        action_stats: dict,     # action min/max/mask
        rel_arm_stats: dict,     # {"left_arm":{min,max}, "right_arm":{min,max}}
        ref_left: np.ndarray,     # (B,dL)
        ref_right: np.ndarray,     # (B,dR)
        dL: int,
        dR: int,
        T_use: Optional[int] = None,     # None -> use all T
) -> np.ndarray:
    """
    SINGLE source of truth for BOTH sim-step and open-loop.

    Steps:
      1) take first 29 dims of na
      2) unnormalize all dims with action_stats + mask
      3) override left/right arm dims with rel_arm_stats (still REL)
      4) rel -> abs by +ref_state (t=0)
      5) return (B,T,32) where:
           - [:29] are valid
           - [29:] are zeros
    """
    na = np.clip(normalized_actions_32, -1.0, 1.0).astype(np.float32)
    assert na.ndim == 3 and na.shape[-1] == 32, f"Expect (B,T,32), got {na.shape}"
    B, T, _ = na.shape
    if T_use is None:
        T_use = T
    T_use = min(T_use, T)

    na29 = na[:, :, :29]     # (B,T,29)
    raw29 = unnormalize_actions_with_mask(na29, action_stats)     # (B,T,29)

    def _override_arm(slot_slice: slice, arm_name: str):
        st = rel_arm_stats[arm_name]
        lo = _as_np_f32(st["min"])
        hi = _as_np_f32(st["max"])

        D_slot = slot_slice.stop - slot_slice.start
        if lo.ndim == 1:
            D = min(D_slot, lo.shape[0])
        elif lo.ndim == 2:
            D = min(D_slot, lo.shape[1])
        else:
            raise ValueError(f"{arm_name} stats shape unsupported: {lo.shape}")

        lo_btd = _stats_to_btd(lo, T_use=T_use, D=D)
        hi_btd = _stats_to_btd(hi, T_use=T_use, D=D)

        raw29[:, :T_use, slot_slice.start:slot_slice.start + D] = _inv_minmax(
            na29[:, :T_use, slot_slice.start:slot_slice.start + D],
            lo_btd,
            hi_btd,
        )

    left_slice = slice(0, dL)
    right_slice = slice(dL, dL + dR)

    _override_arm(left_slice, "left_arm")
    _override_arm(right_slice, "right_arm")

    # rel -> abs
    raw29[:, :T_use, left_slice] += ref_left[:, None, :dL]
    raw29[:, :T_use, right_slice] += ref_right[:, None, :dR]

    out = np.zeros((B, T, 32), dtype=np.float32)
    out[:, :, :29] = raw29
    return out


def resize_pad_to_float01(
    image: np.ndarray,
    image_size: List[int] = [224, 224],
    interpolation: str = "linear",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Geometry: scale longest edge to S, then center pad to SxS.
    Output: float32 HWC in [0,1].
    """
    img = image
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
    else:
        img = img.astype(np.float32) / 255.0

    H, W = img.shape[:2]
    S = int(image_size[0])
    scale = S / max(H, W)
    nh, nw = int(round(H * scale)), int(round(W * scale))

    cv_interp = cv.INTER_LINEAR if interpolation == "linear" else cv.INTER_AREA
    resized = cv.resize(img, (nw, nh), interpolation=cv_interp)

    pad_h, pad_w = S - nh, S - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    fv = float(fill_value)
    padded = cv.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv.BORDER_CONSTANT,
        value=(fv, fv, fv),
    )
    return padded.astype(np.float32)


# ============================================================
# 1) Inference interface (robocasa sim) - aligned to open-loop
# ============================================================


class PolicyWarper:
    """
    Sim interface aligned with open-loop:
      - image preprocessing: resize_pad_to_float01 (shared)
      - state preprocessing: minmax -> [-1,1] (shared)
      - predict payload: {"image":[...], "lang":..., "state":...}
      - action postprocess: postprocess_actions_rel2abs (shared)
    """

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        horizon: int = 0,
        action_ensemble: bool = False,
        action_ensemble_horizon: int = 3,
        adaptive_ensemble_alpha: float = 0.1,
        image_size: list[int] = [224, 224],
        interpolation: str = "linear",
        fill_value: float = 0.0,
        host: str = "0.0.0.0",
        port: int = 10095,
        n_action_steps: int = 2,
    ):
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key

        self.horizon = int(horizon)
        self.image_history = deque(maxlen=self.horizon)
        self.task_description = None

        self.image_size = image_size
        self.fill_value = float(fill_value)
        self.interpolation = interpolation

        self.n_action_steps = int(n_action_steps)

        # ensemble
        self.action_ensemble = bool(action_ensemble)
        self.action_ensembler = (AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha) if action_ensemble else None)

        # stats
        self.action_stats = load_action_stats(policy_ckpt_path, unnorm_key)
        self.state_stats = load_state_stats(policy_ckpt_path, unnorm_key)
        self.rel_arm_stats = load_rel_arm_stats(policy_ckpt_path, unnorm_key)

        # ---- IMPORTANT: define state concat order to match your training ----
        # If your training used different order, change this list accordingly.
        self.state_keys = [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
            "state.waist",
        ]

    def reset(self, task_description: str):
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensembler:
            self.action_ensembler.reset()

    def _prepare_state(self, observations: dict) -> np.ndarray:
        """
        observations contain each key shaped (B,1,Dk). We concat along last dim -> (B,1,D).
        Then normalize to [-1,1] using self.state_stats -> return (B,1,D).
        """
        parts = [observations[k] for k in self.state_keys]     # each (B,1,Dk)
        state = np.concatenate(parts, axis=-1).astype(np.float32)     # (B,1,D)
        # normalize absolute state
        state_norm = normalize_with_mask_minmax_to_minus1_1(state, self.state_stats)     # (B,1,D)
        return state_norm

    def step(self, observations: dict) -> dict:
        task_desc = observations["annotation.human.coarse_action"][0]
        if task_desc != self.task_description:
            self.reset(task_desc)

        # images: (B,1,H,W,3)
        images = observations["video.ego_view"]
        images = [[
            resize_pad_to_float01(
                img,
                image_size=self.image_size,
                interpolation=self.interpolation,
                fill_value=self.fill_value,
            ) for img in sample
        ] for sample in images]
        B = len(images)

        # state: (B,1,D) normalized
        state_norm = self._prepare_state(observations)

        # --- payload: image + lang + state ---
        payload = {
            "examples": [
                {
                    "image": [images[b][0]],     # keep list-of-images as your convention
                    "lang": task_desc,
                    "state": state_norm[b],     # (1,D)
                } for b in range(B)
            ]
        }

        resp = self.client.predict_action(payload)
        data = resp["data"]
        if "normalized_actions" not in data:
            raise RuntimeError(f"response['data'] has no 'normalized_actions'. keys={list(data.keys())}")

        na = np.asarray(data["normalized_actions"], dtype=np.float32)     # (B,T,32)

        # shared postprocess: rel->abs for arms
        dL = observations["state.left_arm"].shape[-1]
        dR = observations["state.right_arm"].shape[-1]
        ref_left = observations["state.left_arm"][:, 0, :dL].astype(np.float32)
        ref_right = observations["state.right_arm"][:, 0, :dR].astype(np.float32)

        raw32 = postprocess_actions_rel2abs(
            normalized_actions_32=na,
            action_stats=self.action_stats,
            rel_arm_stats=self.rel_arm_stats,
            ref_left=ref_left,
            ref_right=ref_right,
            dL=dL,
            dR=dR,
            T_use=None,
        )

        if self.action_ensemble:
            raw32 = np.stack(
                [self.action_ensembler.ensemble_action(raw32[b])[None] for b in range(B)],
                axis=0,
            )

        out = {
            "action.left_arm": raw32[:, :self.n_action_steps, 0:dL],
            "action.right_arm": raw32[:, :self.n_action_steps, dL:dL + dR],
            "action.left_hand": raw32[:, :self.n_action_steps, dL + dR:dL + dR + 6],
            "action.right_hand": raw32[:, :self.n_action_steps, dL + dR + 6:dL + dR + 12],
            "action.waist": raw32[:, :self.n_action_steps, dL + dR + 12:dL + dR + 15],
        }
        return {"actions": out}


# ============================================================
# 2) Open-loop evaluator (optional) - uses SAME preprocess & postprocess
# ============================================================


def build_concat_slices(step_data: dict, keys: List[str]) -> Dict[str, slice]:
    cur = 0
    out = {}
    for k in keys:
        dim = int(step_data[k].shape[1])
        out[k] = slice(cur, cur + dim)
        cur += dim
    return out


def build_obs_from_step_data(step_data: dict, single_ds, img_proc: Optional[Callable] = None) -> dict:
    video_keys = single_ds.modality_keys["video"]
    head = step_data[video_keys[0]][0]
    if img_proc is not None:
        head = img_proc(head)

    state_parts = [step_data[k] for k in single_ds.modality_keys["state"]]
    state = np.concatenate(state_parts, axis=1)
    state0 = state[0].astype(np.float32)

    lang_key = single_ds.modality_keys["language"][0]
    lang = step_data[lang_key][0]
    if isinstance(lang, list):
        lang = lang[0]

    action_slices = build_concat_slices(step_data, single_ds.modality_keys["action"])
    state_slices = build_concat_slices(step_data, single_ds.modality_keys["state"])

    return {
        "observation/image": head,
        "observation/state": state0,
        "prompt": lang,
        "slices/action.left_arm": action_slices["action.left_arm"],
        "slices/action.right_arm": action_slices["action.right_arm"],
        "slices/state.left_arm": state_slices["state.left_arm"],
        "slices/state.right_arm": state_slices["state.right_arm"],
    }


def get_gt_action_chunk_abs(step_data: dict, single_ds) -> np.ndarray:
    action_parts = [step_data[k] for k in single_ds.modality_keys["action"]]
    return np.concatenate(action_parts, axis=1).astype(np.float32)     # (H,D)


def get_gt_action_chunk_rel(step_data: dict, single_ds, obs: dict) -> np.ndarray:
    gt_abs = get_gt_action_chunk_abs(step_data, single_ds)
    a_la: slice = obs["slices/action.left_arm"]
    a_ra: slice = obs["slices/action.right_arm"]
    s_la: slice = obs["slices/state.left_arm"]
    s_ra: slice = obs["slices/state.right_arm"]

    state0 = obs["observation/state"].astype(np.float32)
    ref_left = state0[s_la]
    ref_right = state0[s_ra]

    gt_rel = gt_abs.copy()
    dL = min(a_la.stop - a_la.start, ref_left.shape[0])
    dR = min(a_ra.stop - a_ra.start, ref_right.shape[0])
    gt_rel[:, a_la.start:a_la.start + dL] -= ref_left[:dL][None, :]
    gt_rel[:, a_ra.start:a_ra.start + dR] -= ref_right[:dR][None, :]
    return gt_rel


def get_relative_arm_stats_from_ckpt(norm_stats: dict, unnorm_key: str) -> dict:
    ns = norm_stats[unnorm_key]

    def _pack(k):
        s = ns[k]
        return {"min": _as_np_f32(s["min"]), "max": _as_np_f32(s["max"])}

    return {"left_arm": _pack("relative_action.left_arm"), "right_arm": _pack("relative_action.right_arm")}


def _build_policy_for_openloop(args):
    vla = baseframework.from_pretrained(args.ckpt_path).to("cuda").eval()
    cfg, norm_stats = read_mode_config(Path(args.ckpt_path))
    cfg = OmegaConf.create(cfg)

    ns = norm_stats[args.unnorm_key]
    vla.action_stats = ns["action"]
    vla.state_stats = ns["state"]
    vla.rel_arm_stats = get_relative_arm_stats_from_ckpt(norm_stats, args.unnorm_key)
    return vla, cfg


def infer_openloop(policy, obs: dict) -> Dict:
    """
    Payload now includes state (normalized by min/max to [-1,1]) with no relative ops.
    """
    state0 = obs["observation/state"][None, :]     # (1,D_state)
    state_norm = normalize_with_mask_minmax_to_minus1_1(state0, policy.state_stats)     # (1,D_state)
    state_norm = state_norm[:, None, :]     # (1,1,D_state)  match training typical shape

    D = state_norm.shape[-1]
    if D < 64:
        pad = 64 - D
        state_norm = np.pad(
            state_norm,
            ((0, 0), (0, 0), (0, pad)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        state_norm = state_norm[..., :64]

    fake_data = {
        "image": [obs["observation/image"]],
        "lang": obs["prompt"],
        "state": state_norm[0],     # (1,D_state)
    }

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = policy.predict_action(fake_data)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    na = out["normalized_actions"]     # (1,T,32)

    # ref from state0 slices (robocasa arms)
    s_la: slice = obs["slices/state.left_arm"]
    s_ra: slice = obs["slices/state.right_arm"]

    # arms dims assumed 7; if not, adjust here
    ref_left = state0[:, s_la][:, :7]
    ref_right = state0[:, s_ra][:, :7]

    pred_abs_32 = postprocess_actions_rel2abs(
        normalized_actions_32=na,
        action_stats=policy.action_stats,
        rel_arm_stats=policy.rel_arm_stats,
        ref_left=ref_left,
        ref_right=ref_right,
        dL=7,
        dR=7,
        T_use=None,
    )
    pred_abs_29 = pred_abs_32[0, :, :29]     # (T,29)

    # diagnostic rel
    a_la: slice = obs["slices/action.left_arm"]
    a_ra: slice = obs["slices/action.right_arm"]
    pred_rel_29 = pred_abs_29.copy()
    pred_rel_29[:, a_la] -= ref_left[0, :7][None, :]
    pred_rel_29[:, a_ra] -= ref_right[0, :7][None, :]

    return {"pred_abs": pred_abs_29, "pred_rel": pred_rel_29, "infer_ms": infer_ms}


def rollout_open_loop(single_ds, policy, args, traj_id: int, img_proc: Callable):
    traj_len = int(single_ds.trajectory_lengths[single_ds.get_trajectory_index(traj_id)])
    actual_steps = min(args.steps, traj_len)

    pred_abs_all, gt_abs_all = [], []
    pred_rel_all, gt_rel_all = [], []
    infer_ms_list = []

    H = args.action_horizon
    for t in range(0, actual_steps, H):
        step_data = single_ds.get_step_data(traj_id, t)
        obs = build_obs_from_step_data(step_data, single_ds, img_proc=img_proc)

        pred = infer_openloop(policy, obs)

        gt_abs = get_gt_action_chunk_abs(step_data, single_ds)
        gt_rel = get_gt_action_chunk_rel(step_data, single_ds, obs)

        remain = actual_steps - t
        pred_abs_all.append(pred["pred_abs"][:remain])
        gt_abs_all.append(gt_abs[:remain])

        pred_rel_all.append(pred["pred_rel"][:remain])
        gt_rel_all.append(gt_rel[:remain])

        infer_ms_list.append(pred["infer_ms"])

    pred_abs = np.concatenate(pred_abs_all, axis=0)
    gt_abs = np.concatenate(gt_abs_all, axis=0)
    pred_rel = np.concatenate(pred_rel_all, axis=0)
    gt_rel = np.concatenate(gt_rel_all, axis=0)

    print("mean |pred_abs - pred_rel| =", np.mean(np.abs(pred_abs - pred_rel)))
    print("mean |gt_abs   - gt_rel|   =", np.mean(np.abs(gt_abs - gt_rel)))
    print("mean |(pred_abs - pred_rel) - (gt_abs - gt_rel)| =", np.mean(np.abs((pred_abs - pred_rel) - (gt_abs - gt_rel))))

    return (pred_abs, gt_abs, pred_rel, gt_rel, infer_ms_list)


def plot_action_curves(gt: np.ndarray, pred: np.ndarray, save_path: str, title: str = ""):
    T, D = gt.shape
    fig, axes = plt.subplots(nrows=D, ncols=1, figsize=(10, 2.0 * D))
    if D == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)
    x = np.arange(T)
    for d in range(D):
        ax = axes[d]
        ax.plot(x, gt[:, d], label="gt")
        ax.plot(x, pred[:, d], label="pred")
        ax.set_ylabel(f"dim {d}")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time step")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@dataclass
class OpenLoopArgs:
    ckpt_path: str = "outputs/robocasa_1000_rel/checkpoints/steps_150000_pytorch_model.pt"
    unnorm_key: str = "gr1"
    image_size: int = 224
    image_interpolation: str = "linear"
    image_fill_value: float = 0.0
    steps: int = 200
    action_horizon: int = 16
    traj_id: int = 1


def main_openloop(args: OpenLoopArgs):
    policy, cfg = _build_policy_for_openloop(args)
    print("policy loaded")

    img_proc = lambda img: resize_pad_to_float01(
        img,
        image_size=[args.image_size, args.image_size],
        interpolation=args.image_interpolation,
        fill_value=args.image_fill_value,
    )

    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.task_id = "all"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    single_ds = dataset.datasets[0] if hasattr(dataset, "datasets") else dataset

    traj_id = int(args.traj_id) if args.traj_id >= 0 else int(single_ds.trajectory_ids[0])
    print("Using traj_id:", traj_id)

    pred_abs, gt_abs, pred_rel, gt_rel, infer_ms_list = rollout_open_loop(single_ds, policy, args, traj_id, img_proc)

    mse_abs = np.mean((pred_abs - gt_abs)**2)
    mse_rel = np.mean((pred_rel - gt_rel)**2)
    print(f"ABS MSE={mse_abs:.6f} | REL MSE={mse_rel:.6f}")
    print(f"Infer ms mean={np.mean(infer_ms_list):.2f}, p95={np.percentile(infer_ms_list, 95):.2f}")

    out_dir = Path("outputs/open_loop_eval_abs_rel")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_action_curves(gt_abs, pred_abs, str(out_dir / f"traj{traj_id}_ABS.png"), title="ABS action curves")
    plot_action_curves(gt_rel, pred_rel, str(out_dir / f"traj{traj_id}_REL.png"), title="REL action curves")

    print("Saved to:", out_dir)


if __name__ == "__main__":
    main_openloop(tyro.cli(OpenLoopArgs))
