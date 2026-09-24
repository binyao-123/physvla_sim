#!/usr/bin/env python3
"""PRTS inference adapter for Isaac sim rollout.

Loads the local TeleHuman checkpoint with the cloned PRTS package on
PYTHONPATH. Does not import workspace LeRobot (PRTS wants 0.3.3; this
stack uses a newer tree for π0.5).

Current left6 ckpt observation:
  observation.images.head, observation.images.left_wrist  (no right_wrist)
  observation.state  float32 (7,)  # left 6 joints + left gripper
  prompt: task instruction, e.g. "close the laptop lid."
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# left6 ckpt: overhead + left wrist only. Do not send right_wrist.
_DEFAULT_CAMERA_ORDER = ("head", "left_wrist")
_CAM_MAPPING = {
    "image_0": "camera_top",
    "image_1": "camera_left_wrist",
}

IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
SYSTEM_MESSAGE = "You are a helpful physical assistant."


_DEFAULT_PAD_TOKEN_ID = 151643  # Qwen <|endoftext|>; matches checkpoint top-level pad/bos


def _set_cfg_attr(cfg: object, name: str, value: object) -> None:
    try:
        setattr(cfg, name, value)
    except Exception:
        object.__setattr__(cfg, name, value)


def _sanitize_prts_config(config: object) -> object:
    """Fill fields that transformers 4.57 wrote but 5.5 no longer defaults.

    The checkpoint stores ``pad_token_id`` only on the top-level config. The
    text tower reads ``config.text_config.pad_token_id`` during ``__init__``.
    """

    pad_id = getattr(config, "pad_token_id", None)
    if pad_id is None:
        pad_id = _DEFAULT_PAD_TOKEN_ID
        _set_cfg_attr(config, "pad_token_id", pad_id)

    text = getattr(config, "text_config", None)
    if text is not None:
        text_pad = getattr(text, "pad_token_id", None)
        if text_pad is None:
            text_pad = getattr(text, "bos_token_id", None) or pad_id
            _set_cfg_attr(text, "pad_token_id", text_pad)
        if getattr(text, "bos_token_id", None) is None:
            _set_cfg_attr(text, "bos_token_id", text_pad)
        if getattr(text, "eos_token_id", None) is None:
            _set_cfg_attr(text, "eos_token_id", getattr(config, "eos_token_id", 151645))

    return config


_PRTS_TF_PATCH_VERSION = 3


def _default_rope_inv_freq(config, device=None, seq_len=None, **kwargs):
    """4.57-compatible default RoPE inv_freq (HF later moved this onto the module)."""

    rope_params = getattr(config, "rope_parameters", None)
    base = None
    if isinstance(rope_params, dict):
        base = rope_params.get("rope_theta")
    if base is None:
        base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim)
    )
    return inv_freq, 1.0


def _ensure_rotary_has_default_rope_fn(module: object) -> None:
    """5.5 ``_init_weights`` calls ``module.compute_default_rope_parameters``.

    Checkpoint modeling (transformers 4.57) has no such method. ``inv_freq`` is a
    non-persistent buffer, so from_pretrained always re-inits it after load.
    """

    if "RotaryEmbedding" not in type(module).__name__:
        return
    if hasattr(module, "compute_default_rope_parameters"):
        return
    module.compute_default_rope_parameters = _default_rope_inv_freq


def _legacy_kv_pairs_from_cache(past_key_values: object) -> object:
    """Convert a 5.5 Cache to the 4.57 ``(K, V)`` list that PRTS indexes.

    transformers 4.57.3 ``Cache.__getitem__`` returned
    ``(self.layers[i].keys, self.layers[i].values)``. 5.5 dropped that method
    (HF cache docs now say use ``cache.layers[i].keys`` / ``.values``). Tensor
    layout is unchanged: ``[batch, num_kv_heads, seq, head_dim]``.
    """

    if isinstance(past_key_values, (list, tuple)):
        return past_key_values
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return past_key_values
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise TypeError(
                f"PRTS KV cache layer {idx} is empty "
                f"(keys={type(keys).__name__}, values={type(values).__name__})"
            )
        pairs.append((keys, values))
    return pairs


def _restore_cache_getitem() -> None:
    """Restore 4.57.3 ``Cache.__getitem__`` on transformers 5.5 Cache.

    Official 4.57.3 source (the version this checkpoint was saved with)::

        def __getitem__(self, layer_idx):
            if layer_idx < len(self.layers):
                return self.layers[layer_idx].keys, self.layers[layer_idx].values
            raise KeyError(...)

    5.5 still has ``len(cache)`` and ``cache.layers``, and DynamicCache.__iter__
    now yields 3-tuples ``(keys, values, sliding)``. Do not replace ``__iter__``.
    """

    import transformers.cache_utils as cache_utils

    cache_cls = cache_utils.Cache
    existing = cache_cls.__dict__.get("__getitem__")
    if getattr(existing, "_prts_cache_getitem", False):
        return
    if existing is not None:
        return

    def _getitem(self, layer_idx: int):
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        raise KeyError(
            f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
        )

    _getitem._prts_cache_getitem = True
    cache_cls.__getitem__ = _getitem


def _wrap_dit_predict_action(model: object) -> None:
    """Make DiT ``predict_action`` accept a 5.5 DynamicCache.

    Converts to a list of ``(K, V)`` before the 4.57 checkpoint code runs
    ``past_key_values[0][0]`` / ``_kv_pairs_from_past_key_values``.
    """

    head = getattr(model, "dit_action_head", None)
    if head is None:
        return
    original = head.predict_action
    if getattr(original, "_prts_cache_compat", False):
        return

    logged = {"done": False}

    def _predict_action_compat(past_key_values, *args, **kwargs):
        converted = _legacy_kv_pairs_from_cache(past_key_values)
        if (
            not logged["done"]
            and isinstance(converted, list)
            and converted
            and isinstance(converted[0], tuple)
        ):
            k0, v0 = converted[0]
            print(
                f"[PRTS] dit kv layers={len(converted)} "
                f"k0={tuple(k0.shape)} v0={tuple(v0.shape)}",
                flush=True,
            )
            logged["done"] = True
        return original(converted, *args, **kwargs)

    _predict_action_compat._prts_cache_compat = True
    head.predict_action = _predict_action_compat


def _patch_transformers_for_prts() -> None:
    """PRTS checkpoints were saved with transformers 4.57.3.

    Isaac currently ships 5.5.x. Official 5.5 Qwen3-VL and HF #39831 moved
    default RoPE off ``ROPE_INIT_FUNCTIONS`` onto
    ``RotaryEmbedding.compute_default_rope_parameters``. Loading a 4.57
    remote-code checkpoint then dies in ``_initialize_missing_keys``.
    """

    import transformers.modeling_rope_utils as rope_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.utils.generic as generic

    if getattr(generic.check_model_inputs, "_prts_compat_version", 0) >= _PRTS_TF_PATCH_VERSION:
        return

    original = generic.check_model_inputs

    def compat(func=None, *args, **kwargs):
        if func is None or not callable(func):
            def deco(wrapped):
                return original(wrapped)

            return deco
        return original(func)

    compat._prts_compat = True
    compat._prts_compat_version = _PRTS_TF_PATCH_VERSION
    generic.check_model_inputs = compat

    # 4.57 stored rope_type="default" in ROPE_INIT_FUNCTIONS; 5.5 dropped that key.
    rope_utils.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_inv_freq)

    original_tied = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys
    if not getattr(original_tied, "_prts_tied_compat", False):

        def _tied_compat(self, all_submodels=False):
            keys = getattr(type(self), "_tied_weights_keys", None)
            if isinstance(keys, list):
                converted = {}
                for key in keys:
                    if key == "lm_head.weight":
                        converted[key] = "language_model.embed_tokens.weight"
                    else:
                        converted[key] = key
                type(self)._tied_weights_keys = converted
            return original_tied(self, all_submodels=all_submodels)

        _tied_compat._prts_tied_compat = True
        modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = _tied_compat

    original_init_weights = modeling_utils.PreTrainedModel._init_weights
    if not getattr(original_init_weights, "_prts_rope_compat", False):

        def _init_weights_compat(self, module):
            _ensure_rotary_has_default_rope_fn(module)
            return original_init_weights(self, module)

        _init_weights_compat._prts_rope_compat = True
        modeling_utils.PreTrainedModel._init_weights = _init_weights_compat

    _restore_cache_getitem()


def _ensure_prts_on_path(prts_root: Path) -> None:
    root = str(prts_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import colorama  # noqa: F401
    except ImportError:
        fake = types.ModuleType("colorama")

        class _C:
            BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
            RESET = ""

        fake.Fore = _C()
        fake.Style = type("S", (), {"NORMAL": "", "BRIGHT": "", "RESET_ALL": ""})()
        fake.init = lambda **_kwargs: None
        sys.modules["colorama"] = fake


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Qwen-VL smart resize (qwen_vl_utils / PRTS get_image_info)."""

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


def _chw01_to_pil(image: torch.Tensor) -> Image.Image:
    x = image.detach()
    if x.dim() == 4:
        x = x.squeeze(0)
    if x.dim() != 3:
        raise ValueError(f"expected CHW image, got {tuple(x.shape)}")
    if x.shape[0] in (3, 4):
        x = x[:3]
    else:
        raise ValueError(f"unsupported image layout {tuple(x.shape)}")
    x = x.to(dtype=torch.float32).clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _camera_prompt_name(raw_name: str) -> str:
    if raw_name in _CAM_MAPPING:
        name = _CAM_MAPPING[raw_name]
    elif "image" in raw_name:
        name = raw_name.replace("image", "camera")
    elif "cam" in raw_name:
        name = raw_name.replace("cam", "camera")
    else:
        name = "camera_" + raw_name
    return name.replace("_", " ").capitalize()


def _load_quantile_stats(model_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    stats_path = model_dir / "statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    raw = payload.get("stats") or {}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key in ("observation.state", "action"):
        if key not in raw:
            raise KeyError(f"{stats_path} missing stats for '{key}'")
        out[key] = {
            name: torch.tensor(raw[key][name], dtype=torch.float32)
            for name in ("q01", "q99")
        }
    return out


def _quantile_normalize(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01 = q01.to(device=x.device, dtype=x.dtype)
    q99 = q99.to(device=x.device, dtype=x.dtype)
    return (x - q01) / (q99 - q01 + 1e-8) * 2.0 - 1.0


def _quantile_unnormalize(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01 = q01.to(device=x.device, dtype=x.dtype)
    q99 = q99.to(device=x.device, dtype=x.dtype)
    return (x + 1.0) / 2.0 * (q99 - q01) + q01


class PRTSRuntime:
    """Closed-loop PRTS policy: obs dict → (1, action_dim) absolute action."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        prts_root: str | Path,
        device: torch.device | str = "cuda",
        action_dim: int = 7,
        state_relative_action: bool = True,
        action_exec_horizon: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_dir = Path(model_path).expanduser().resolve()
        if not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"PRTS checkpoint missing config.json: {self.model_dir}")
        self.prts_root = Path(prts_root).expanduser().resolve()
        if not (self.prts_root / "prts").is_dir():
            raise FileNotFoundError(f"PRTS repo not found at {self.prts_root}")

        _ensure_prts_on_path(self.prts_root)
        _patch_transformers_for_prts()

        self.device = torch.device(device)
        self.dtype = dtype
        self.action_dim = int(action_dim)
        self.state_relative_action = bool(state_relative_action)
        self.action_exec_horizon = max(1, int(action_exec_horizon))
        self._action_queue: list[torch.Tensor] = []
        self._infer_calls = 0

        from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer

        print(f"[PRTS] loading tokenizer/image processor from {self.model_dir}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=True
        )
        size = getattr(self.image_processor, "size", None)
        min_pixels = getattr(self.image_processor, "min_pixels", None)
        max_pixels = getattr(self.image_processor, "max_pixels", None)
        if min_pixels is None and size is not None:
            min_pixels = getattr(size, "shortest_edge", None)
        if max_pixels is None and size is not None:
            max_pixels = getattr(size, "longest_edge", None)
        self.min_pixels = int(min_pixels or 65536)
        self.max_pixels = int(max_pixels or 147456)
        self.merge_size = int(getattr(self.image_processor, "merge_size", 2))
        self.patch_size = int(getattr(self.image_processor, "patch_size", 16))
        self.resize_factor = self.patch_size * self.merge_size

        print(f"[PRTS] loading model from {self.model_dir}", flush=True)
        config = AutoConfig.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=True
        )
        config = _sanitize_prts_config(config)
        load_kwargs: dict[str, Any] = {
            "config": config,
            "local_files_only": True,
            "trust_remote_code": True,
        }
        # transformers 5.5 prefers dtype=; 4.57 used torch_dtype=.
        load_kwargs["dtype"] = self.dtype
        try:
            self.model = AutoModel.from_pretrained(self.model_dir, **load_kwargs)
        except TypeError:
            load_kwargs.pop("dtype", None)
            load_kwargs["torch_dtype"] = self.dtype
            self.model = AutoModel.from_pretrained(self.model_dir, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        # Training used FlashAttention-3; Isaac has neither FA3 nor the custom
        # kernels. 5.5 create_causal_mask indexes ALL_MASK_ATTENTION_FUNCTIONS
        # by _attn_implementation and can fail/hang if that key is missing.
        attn_impl = "sdpa"
        try:
            self.model.set_attn_implementation(attn_impl)
        except Exception as exc:
            print(f"[PRTS] set_attn_implementation({attn_impl}) failed: {exc}", flush=True)
        for cfg_obj in (
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "text_config", None),
            getattr(getattr(self.model, "config", None), "vision_config", None),
        ):
            if cfg_obj is not None:
                _set_cfg_attr(cfg_obj, "_attn_implementation", attn_impl)
        print(
            f"[PRTS] attn={getattr(self.model.config, '_attn_implementation', '?')}",
            flush=True,
        )
        _wrap_dit_predict_action(self.model)

        self.norm_stats = _load_quantile_stats(self.model_dir)
        stats_dim = int(self.norm_stats["observation.state"]["q01"].numel())
        if self.action_dim != stats_dim:
            print(
                f"[PRTS] action_dim={self.action_dim} != statistics.json dim={stats_dim}; "
                f"using {stats_dim}",
                flush=True,
            )
            self.action_dim = stats_dim
        embodiment = getattr(self.model.config, "embodiment_tag", "piper_left6")
        from prts.data.embodiment_tag import EMBODIMENT_CONFIGS, EmbodimentConfig

        # Training tag for this left-6 ckpt is not in the upstream registry yet.
        if "piper_left6" not in EMBODIMENT_CONFIGS:
            EMBODIMENT_CONFIGS["piper_left6"] = EmbodimentConfig(
                name="piper_left6",
                # joints 1–6 relative, gripper absolute (lid task gripper is always 0)
                delta_action_mask=[True] * 6 + [False] * 1,
                state_relative=True,
            )
        ecfg = EMBODIMENT_CONFIGS.get(embodiment)
        if ecfg is None:
            raise ValueError(f"unknown embodiment_tag={embodiment}")
        mask = list(ecfg.delta_action_mask)[: self.action_dim]
        if len(mask) < self.action_dim:
            mask.extend([False] * (self.action_dim - len(mask)))
        self.delta_mask = torch.tensor(mask, dtype=torch.bool)
        n_params = sum(p.numel() for p in self.model.parameters())
        dit = getattr(self.model, "dit_action_head", None)
        dit_layers = getattr(dit, "num_dit_layers", None)
        text_layers = getattr(getattr(self.model.config, "text_config", None), "num_hidden_layers", None)
        print(
            f"[PRTS] ready: params={n_params:,} action_dim={self.action_dim} "
            f"chunk={getattr(self.model.config, 'action_chunk_size', '?')} "
            f"denoise={getattr(self.model.config, 'num_denoise_steps', '?')} "
            f"embodiment={embodiment} state_relative={self.state_relative_action} "
            f"exec_horizon={self.action_exec_horizon} "
            f"pixels=[{self.min_pixels},{self.max_pixels}] "
            f"dit_layers={dit_layers} text_layers={text_layers} "
            f"cameras={'+'.join(_DEFAULT_CAMERA_ORDER)}",
            flush=True,
        )

    def reset(self) -> None:
        self._action_queue.clear()

    def select_action(self, obs: dict[str, Any]) -> torch.Tensor:
        if not self._action_queue:
            chunk = self.infer_chunk(obs)
            horizon = min(self.action_exec_horizon, int(chunk.shape[0]))
            self._action_queue = [chunk[i].unsqueeze(0) for i in range(horizon)]
        return self._action_queue.pop(0)

    @torch.no_grad()
    def infer_chunk(self, obs: dict[str, Any]) -> torch.Tensor:
        """Return absolute actions of shape (T, action_dim) on CPU float32."""

        raw_state = obs["observation.state"]
        if isinstance(raw_state, torch.Tensor):
            raw_state = raw_state.detach().float()
        else:
            raw_state = torch.as_tensor(raw_state, dtype=torch.float32)
        if raw_state.dim() == 1:
            raw_state = raw_state.unsqueeze(0)
        raw_state = raw_state[..., : self.action_dim]

        images, camera_names = self._collect_images(obs)
        processed = [self._prepare_image(im) for im in images]
        vision = self.image_processor(
            images=processed,
            do_resize=False,
            return_tensors="pt",
        )
        pixel_values = vision["pixel_values"].to(self.device, dtype=self.dtype)
        image_grid_thw = vision["image_grid_thw"].to(self.device)

        norm_state = _quantile_normalize(
            raw_state,
            self.norm_stats["observation.state"]["q01"][: self.action_dim],
            self.norm_stats["observation.state"]["q99"][: self.action_dim],
        )
        prompt = self._build_prompt(obs, camera_names, norm_state[0], image_grid_thw)
        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        chunk_size = int(self.model.config.action_chunk_size)
        max_action_dim = int(self.model.config.max_action_dim)
        action_dof_mask = torch.zeros(1, chunk_size, max_action_dim, device=self.device)
        action_dof_mask[..., : self.action_dim] = 1.0

        self._infer_calls += 1
        log_infer = self._infer_calls == 1
        if log_infer:
            print(
                f"[PRTS] infer start tokens={tuple(input_ids.shape)} "
                f"pixels={tuple(pixel_values.shape)} grid={tuple(image_grid_thw.shape)}",
                flush=True,
            )
        try:
            outputs, _ = self.model.sample_actions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                action_dof_mask=action_dof_mask,
            )
        except Exception:
            import traceback

            print("[ERROR] PRTS sample_actions failed; traceback follows", flush=True)
            traceback.print_exc()
            raise
        if log_infer:
            print(f"[PRTS] infer done actions={tuple(outputs.shape)}", flush=True)
        pred = outputs[..., : self.action_dim].squeeze(0).float()
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        actions = _quantile_unnormalize(
            pred,
            self.norm_stats["action"]["q01"][: self.action_dim],
            self.norm_stats["action"]["q99"][: self.action_dim],
        )
        if self.state_relative_action:
            state = raw_state.to(device=actions.device, dtype=actions.dtype)
            if state.dim() == 2 and actions.dim() == 2:
                state = state
            mask = self.delta_mask.to(device=actions.device)
            actions = actions.clone()
            actions[..., : mask.numel()] = torch.where(
                mask,
                actions[..., : mask.numel()] + state[..., : mask.numel()],
                actions[..., : mask.numel()],
            )
        # left6: only left 7-D is valid; gripper channel is always closed.
        if actions.shape[-1] >= 7:
            actions = actions.clone()
            actions[..., 6] = 0.0
        return actions.cpu()

    def _collect_images(self, obs: dict[str, Any]) -> tuple[list[torch.Tensor], list[str]]:
        keyed: dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            if not str(key).startswith("observation.images."):
                continue
            keyed[str(key).split(".")[-1]] = value
        names: list[str] = []
        images: list[torch.Tensor] = []
        for name in _DEFAULT_CAMERA_ORDER:
            if name in keyed:
                names.append(name)
                images.append(keyed[name])
        extra = [name for name in keyed if name not in names]
        if extra:
            print(f"[PRTS] ignoring extra cameras: {extra}", flush=True)
        if not images:
            raise ValueError(
                "observation has no observation.images.* keys matching "
                f"{_DEFAULT_CAMERA_ORDER}"
            )
        return images, names

    def _prepare_image(self, image: torch.Tensor) -> Image.Image:
        pil = _chw01_to_pil(image)
        width, height = pil.size
        new_h, new_w = smart_resize(
            height,
            width,
            factor=self.resize_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        if (new_h, new_w) != (height, width):
            pil = pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return pil

    def _build_prompt(
        self,
        obs: dict[str, Any],
        camera_names: list[str],
        norm_state: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> str:
        image_block = "".join(
            f"{_camera_prompt_name(name)}: {VISION_START_TOKEN}{IMAGE_PLACEHOLDER_TOKEN}{VISION_END_TOKEN}\n"
            for name in camera_names
        )
        discretized = ((norm_state + 1.0) / 2.0 * 1000.0).to(torch.int32).clamp(-1, 1000)
        state_block = (
            "Proprioception (normalized to 0-1000 scale): "
            + " ".join(str(int(v)) for v in discretized.tolist())
            + "\n"
        )
        task = obs.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        task_text = str(task).strip()
        user_content = (
            f"{image_block}{state_block}"
            f"Instruction: {task_text}\n"
            "Predict the next action chunk in low-level robotics action format."
        )
        text = (
            f"{IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{IM_END_TOKEN}\n"
            f"{IM_START_TOKEN}user\n{user_content}{IM_END_TOKEN}\n"
            f"{IM_START_TOKEN}assistant\n"
        )
        merge_length = self.merge_size**2
        for grid in image_grid_thw:
            n_tokens = int(grid.prod().item()) // merge_length
            text = text.replace(IMAGE_PLACEHOLDER_TOKEN, "<|placeholder|>" * n_tokens, 1)
        return text.replace("<|placeholder|>", IMAGE_PLACEHOLDER_TOKEN)
