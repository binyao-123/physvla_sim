#!/usr/bin/env python3
"""Websocket client runtime for a remote PRTS policy server.

Same interface as ``prts_runtime.PRTSRuntime`` (``reset`` / ``select_action``)
so ``prts_policy_rollout.py`` can swap between local and remote inference.

Wire format required by the A100 ``websocket_policy_server``:
  observation/head        uint8   (224, 224, 3)  HWC RGB
  observation/left_wrist  uint8   (224, 224, 3)  HWC RGB
  observation/state       float32 (7,)           left 6 joints + gripper, rad
  prompt                  str                    e.g. "close the laptop lid."

Do not send observation/right_wrist: it adds vision tokens the left6 model
never saw in training. Dict insertion order is the camera order.

The server already applies QUANTILE unnormalize and the state-relative
inverse, so ``actions`` are absolute joint angles — do not add state back.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from episode_config import INFER_STREAM, derive_episode_seed

_CAMERA_ORDER = ("head", "left_wrist")


def _ensure_openpi_client_on_path(prts_root: Path) -> None:
    src = (Path(prts_root).expanduser().resolve() / "third_party" / "openpi-client" / "src")
    if not (src / "openpi_client").is_dir():
        raise FileNotFoundError(f"openpi_client not found under {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) float [0,1] / uint8 → (H,W,3) uint8 HWC."""

    x = image.detach()
    if x.dim() == 4:
        x = x[0]
    if x.dim() != 3:
        raise ValueError(f"expected CHW image, got {tuple(x.shape)}")
    if x.shape[0] in (3, 4):
        x = x[:3].permute(1, 2, 0)
    elif x.shape[-1] in (3, 4):
        x = x[..., :3]
    else:
        raise ValueError(f"unsupported image layout {tuple(x.shape)}")
    x = x.to(dtype=torch.float32).cpu()
    if float(x.max()) <= 1.5:
        x = x * 255.0
    return x.round().clamp(0, 255).to(torch.uint8).numpy()


class PRTSRemoteRuntime:
    """Closed-loop PRTS policy over websocket: obs dict → (1, action_dim)."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        prts_root: str | Path,
        action_dim: int = 7,
        action_exec_horizon: int = 50,
        api_key: str | None = None,
        max_reconnect_attempts: int = 6,
        infer_timeout_s: float = 120.0,
    ) -> None:
        _ensure_openpi_client_on_path(Path(prts_root))
        from openpi_client import websocket_client_policy

        self._policy_module = websocket_client_policy
        self._api_key = api_key
        self._max_reconnect_attempts = max(0, int(max_reconnect_attempts))
        self._infer_timeout_s = float(infer_timeout_s)
        self.host = host
        self.port = int(port)
        self.action_dim = int(action_dim)
        self.action_exec_horizon = max(1, int(action_exec_horizon))
        self._action_queue: list[torch.Tensor] = []
        self._infer_calls = 0
        self._episode_seed: int | None = None
        self._infer_index = 0

        print(f"[PRTS] connecting to ws://{host}:{self.port}", flush=True)
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=self.port, api_key=api_key
        )
        meta = self.client.get_server_metadata()
        print(
            f"[PRTS] remote ready: action_dim={self.action_dim} "
            f"exec_horizon={self.action_exec_horizon} "
            f"cameras={'+'.join(_CAMERA_ORDER)} server_meta={meta}",
            flush=True,
        )

    def _reconnect(self) -> None:
        self.client = self._policy_module.WebsocketClientPolicy(
            host=self.host, port=self.port, api_key=self._api_key
        )
        self.client.get_server_metadata()

    def _infer_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``client.infer`` but with a read deadline.

        openpi's ``infer`` calls ``recv()`` with no timeout. When the SSH tunnel dies
        mid-request the socket neither errors nor delivers, so the call blocks forever and
        the reconnect path below never runs -- a collection run silently stops instead of
        recovering. A deadline turns that hang into an exception we can retry.
        """

        ws = getattr(self.client, "_ws", None)
        packer = getattr(self.client, "_packer", None)
        if ws is None or packer is None:  # openpi internals moved; fall back
            return self.client.infer(payload)

        from openpi_client import msgpack_numpy

        ws.send(packer.pack(payload))
        response = ws.recv(timeout=self._infer_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def _infer_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Survive a dropped websocket instead of losing the whole collection run.

        The SSH tunnel to the policy server goes down with the local network, and a
        multi-hour collection should not die with it. Retries are safe: inference is
        stateless server-side, and the payload carries its own seed, so a resent request
        returns the same chunk it would have the first time.
        """

        last_exc: Exception | None = None
        for attempt in range(self._max_reconnect_attempts + 1):
            try:
                return self._infer_once(payload)
            except Exception as exc:  # websocket errors are not one exception type
                last_exc = exc
                if attempt == self._max_reconnect_attempts:
                    break
                delay = min(30.0, 2.0 * (2**attempt))
                print(
                    f"[WARN] PRTS infer failed ({type(exc).__name__}: {exc}); "
                    f"reconnecting in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{self._max_reconnect_attempts})",
                    flush=True,
                )
                time.sleep(delay)
                try:
                    self._reconnect()
                    print("[INFO] PRTS reconnected", flush=True)
                except Exception as reconnect_exc:
                    print(f"[WARN] reconnect failed: {reconnect_exc}", flush=True)
        raise RuntimeError(
            f"PRTS remote infer failed after {self._max_reconnect_attempts} reconnect attempts"
        ) from last_exc

    def reset(self) -> None:
        self._action_queue.clear()
        self._infer_index = 0

    def set_episode_seed(self, seed: int | None) -> None:
        """Make this episode's inference noise reproducible.

        The server's flow matching starts from Gaussian noise, so the same
        observation yields a different chunk on every call unless a seed is sent.
        """

        self._episode_seed = None if seed is None else int(seed)
        self._infer_index = 0

    def select_action(self, obs: dict[str, Any]) -> torch.Tensor:
        if not self._action_queue:
            chunk = self.infer_chunk(obs)
            horizon = min(self.action_exec_horizon, int(chunk.shape[0]))
            self._action_queue = [chunk[i].unsqueeze(0) for i in range(horizon)]
        return self._action_queue.pop(0)

    def infer_chunk(self, obs: dict[str, Any]) -> torch.Tensor:
        """Return absolute actions of shape (T, action_dim) on CPU float32."""

        state = obs["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.detach().float().reshape(-1)[: self.action_dim].cpu()
        if state.numel() != self.action_dim:
            raise ValueError(
                f"state has {state.numel()} dims, server expects {self.action_dim}"
            )

        task = obs.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""

        # Insertion order is the camera order the server prompts with.
        payload: dict[str, Any] = {}
        for name in _CAMERA_ORDER:
            key = f"observation.images.{name}"
            if key not in obs:
                raise KeyError(f"observation missing {key}")
            payload[f"observation/{name}"] = _to_uint8_hwc(obs[key])
        payload["observation/state"] = state.numpy().astype(np.float32)
        payload["prompt"] = str(task).strip()
        if self._episode_seed is not None:
            payload["seed"] = derive_episode_seed(
                self._episode_seed, self._infer_index, INFER_STREAM
            )
        self._infer_index += 1

        self._infer_calls += 1
        log_infer = self._infer_calls == 1
        if log_infer:
            shapes = {k: getattr(v, "shape", v) for k, v in payload.items()}
            print(f"[PRTS] infer start {shapes}", flush=True)

        t0 = time.perf_counter()
        try:
            out = self._infer_with_retry(payload)
        except Exception:
            import traceback

            print("[ERROR] PRTS remote infer failed; traceback follows", flush=True)
            traceback.print_exc()
            raise
        rtt_ms = (time.perf_counter() - t0) * 1000.0

        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if log_infer:
            print(
                f"[PRTS] infer done actions={actions.shape} rtt={rtt_ms:.0f}ms "
                f"server={out.get('server_timing')}",
                flush=True,
            )

        chunk = torch.from_numpy(actions[:, : self.action_dim].copy())
        # Gripper channel is untrained (q01 == q99 == 0); never drive it.
        if chunk.shape[-1] >= 7:
            chunk[:, 6] = 0.0
        return chunk
