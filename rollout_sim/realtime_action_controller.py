"""Real-time action chunk fusion and smoothing for PI05 simulation rollout."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class OneEuroFilter:
    """Scalar One Euro filter driven by simulation time."""

    def __init__(
        self,
        t0: float,
        x0: float,
        *,
        min_cutoff: float,
        beta: float,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    @staticmethod
    def _smoothing_factor(elapsed: float, cutoff: float) -> float:
        ratio = 2.0 * math.pi * cutoff * elapsed
        return ratio / (ratio + 1.0)

    def __call__(self, timestamp: float, value: float) -> float:
        elapsed = float(timestamp) - self.t_prev
        if elapsed <= 0.0:
            return self.x_prev

        derivative = (float(value) - self.x_prev) / elapsed
        derivative_alpha = self._smoothing_factor(elapsed, self.d_cutoff)
        derivative_hat = (
            derivative_alpha * derivative + (1.0 - derivative_alpha) * self.dx_prev
        )
        cutoff = self.min_cutoff + self.beta * abs(derivative_hat)
        value_alpha = self._smoothing_factor(elapsed, cutoff)
        value_hat = value_alpha * float(value) + (1.0 - value_alpha) * self.x_prev

        self.x_prev = value_hat
        self.dx_prev = derivative_hat
        self.t_prev = float(timestamp)
        return value_hat


@dataclass
class ActionChunk:
    start_step: int
    actions: torch.Tensor


class RealtimeActionController:
    """Fuse overlapping chunks, filter targets, and enforce joint speed limits."""

    def __init__(
        self,
        *,
        control_hz: float,
        replan_hz: float,
        ensemble_k: float = 0.0625,
        max_joint_speed_rad_s: float = 0.9,
        gripper_indices: tuple[int, ...] = (6, 13),
    ) -> None:
        if control_hz <= 0 or replan_hz <= 0:
            raise ValueError("control_hz and replan_hz must be positive")
        if replan_hz > control_hz:
            raise ValueError("replan_hz cannot exceed control_hz")
        if max_joint_speed_rad_s <= 0:
            raise ValueError("max_joint_speed_rad_s must be positive")

        self.control_hz = float(control_hz)
        self.replan_hz = float(replan_hz)
        self.replan_interval = max(1, int(round(self.control_hz / self.replan_hz)))
        self.ensemble_k = float(ensemble_k)
        self.max_joint_delta = float(max_joint_speed_rad_s) / self.control_hz
        self.gripper_indices = tuple(gripper_indices)
        self._chunks: list[ActionChunk] = []
        self._fast_filters: list[OneEuroFilter | None] = []
        self._controller_filters: list[OneEuroFilter | None] = []
        self._last_target: torch.Tensor | None = None

    def reset(self, current_state: torch.Tensor) -> None:
        state = current_state.detach().flatten().to(dtype=torch.float32)
        self._chunks.clear()
        self._last_target = state.clone()
        self._fast_filters = []
        self._controller_filters = []
        for index, value in enumerate(state):
            if index in self.gripper_indices:
                self._fast_filters.append(None)
                self._controller_filters.append(None)
                continue
            initial = float(value)
            self._fast_filters.append(
                OneEuroFilter(0.0, initial, min_cutoff=0.5, beta=15.0)
            )
            self._controller_filters.append(
                OneEuroFilter(0.0, initial, min_cutoff=0.1, beta=0.1)
            )

    def should_replan(self, control_step: int) -> bool:
        return not self._chunks or (control_step - 1) % self.replan_interval == 0

    def add_chunk(self, control_step: int, actions: torch.Tensor) -> None:
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk shape (steps, dof), got {actions.shape}")
        self._chunks.append(
            ActionChunk(
                start_step=int(control_step),
                actions=actions.detach().to(dtype=torch.float32),
            )
        )
        self._chunks = [
            chunk
            for chunk in self._chunks
            if control_step - chunk.start_step < chunk.actions.shape[0]
        ]

    def _ensemble_for_step(self, control_step: int) -> torch.Tensor | None:
        candidates: list[torch.Tensor] = []
        for chunk in self._chunks:
            action_index = control_step - chunk.start_step
            if 0 <= action_index < chunk.actions.shape[0]:
                candidates.append(chunk.actions[action_index])
        if not candidates:
            return None

        stacked = torch.stack(candidates)
        chunk_age = torch.arange(
            len(candidates) - 1,
            -1,
            -1,
            dtype=stacked.dtype,
            device=stacked.device,
        )
        weights = torch.exp(-self.ensemble_k * chunk_age)
        weights = weights / weights.sum()
        action = (stacked * weights.unsqueeze(1)).sum(dim=0)
        if self.gripper_indices:
            action[list(self.gripper_indices)] = stacked[-1, list(self.gripper_indices)]
        return action

    def action_for_step(
        self,
        control_step: int,
        current_state: torch.Tensor,
    ) -> torch.Tensor | None:
        action = self._ensemble_for_step(control_step)
        if action is None:
            return None

        state = current_state.detach().flatten().to(device=action.device, dtype=action.dtype)
        if self._last_target is None or self._last_target.shape != action.shape:
            self.reset(state)
            return None

        timestamp = float(control_step) / self.control_hz
        filtered = action.clone()
        for index in range(action.shape[0]):
            if index in self.gripper_indices:
                continue
            fast_filter = self._fast_filters[index]
            controller_filter = self._controller_filters[index]
            if fast_filter is None or controller_filter is None:
                continue
            fast_value = fast_filter(timestamp, float(action[index]))
            filtered[index] = controller_filter(timestamp, fast_value)

        previous = self._last_target.to(device=filtered.device, dtype=filtered.dtype)
        arm_mask = torch.ones(filtered.shape[0], dtype=torch.bool, device=filtered.device)
        if self.gripper_indices:
            arm_mask[list(self.gripper_indices)] = False
        delta = filtered[arm_mask] - previous[arm_mask]
        filtered[arm_mask] = previous[arm_mask] + delta.clamp(
            min=-self.max_joint_delta,
            max=self.max_joint_delta,
        )
        self._last_target = filtered.detach().clone()
        return filtered
