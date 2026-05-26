"""HDF5 episode recording via Isaac Lab EpisodeData + HDF5DatasetFileHandler."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import h5py
import torch
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


class OfficialEpisodeCollector:
    def __init__(self, dataset_file: str, env_name: str, num_demos: int = 0):
        self.dataset_file = self._make_session_dataset_file(dataset_file)
        self.env_name = env_name
        self.num_demos = num_demos

        output_dir = os.path.dirname(self.dataset_file) or "."
        output_name = os.path.splitext(os.path.basename(self.dataset_file))[0]
        os.makedirs(output_dir, exist_ok=True)

        self._dataset_handler = HDF5DatasetFileHandler()
        dataset_stem_path = os.path.join(output_dir, output_name)
        dataset_hdf5_path = f"{dataset_stem_path}.hdf5"
        if os.path.exists(dataset_hdf5_path):
            self._dataset_handler._hdf5_file_stream = h5py.File(dataset_hdf5_path, "a")
            self._dataset_handler._hdf5_data_group = self._dataset_handler._hdf5_file_stream.require_group(
                "data"
            )

            demo_ids: list[int] = []
            for name in self._dataset_handler._hdf5_data_group.keys():
                m = re.fullmatch(r"demo_(\d+)", name)
                if m is not None:
                    demo_ids.append(int(m.group(1)))

            next_demo_id = (max(demo_ids) + 1) if demo_ids else 0
            self._dataset_handler._demo_count = next_demo_id
            self.existing_episode_count = len(demo_ids)

            try:
                existing_env_name = self._dataset_handler.get_env_name()
            except Exception:
                existing_env_name = None

            if existing_env_name is None:
                self._dataset_handler.set_env_name(self.env_name)
                existing_env_name = self.env_name

            if existing_env_name != self.env_name:
                print(
                    f"[WARN] Existing dataset env_name='{existing_env_name}' != requested env_name='{self.env_name}'. "
                    "Appending anyway."
                )

            print(
                f"[INFO] Append mode enabled: {dataset_hdf5_path} "
                f"(existing episodes: {self.existing_episode_count}, next demo id: {next_demo_id})"
            )
        else:
            self._dataset_handler.create(dataset_stem_path, env_name=self.env_name)
            self.existing_episode_count = 0
            print(f"[INFO] Create mode enabled: {dataset_hdf5_path}")

        self._episode = EpisodeData()
        self.exported_successful_episode_count = 0
        self.exported_failed_episode_count = 0

    @staticmethod
    def _make_session_dataset_file(dataset_file: str) -> str:
        output_dir = os.path.dirname(dataset_file) or "."
        output_name = os.path.splitext(os.path.basename(dataset_file))[0]
        session_dir = os.path.join(output_dir, output_name)
        utc8 = timezone(timedelta(hours=8))
        timestamp = datetime.now(utc8).strftime("%Y%m%d_%H%M%S")
        return os.path.join(session_dir, f"{output_name}_{timestamp}.hdf5")

    def _prepare_episode_for_export(self):
        if hasattr(self._episode, "pre_export") and callable(self._episode.pre_export):
            self._episode.pre_export()
            return

        def _stack_leaf_lists(node):
            for key, value in node.items():
                if isinstance(value, list):
                    if len(value) > 0 and torch.is_tensor(value[0]):
                        node[key] = torch.stack(value)
                elif isinstance(value, dict):
                    _stack_leaf_lists(value)

        _stack_leaf_lists(self._episode.data)

    def reset_episode(self):
        self._episode = EpisodeData()

    def has_data(self) -> bool:
        return not self._episode.is_empty()

    def set_initial_state(self, initial_state: dict[str, torch.Tensor]):
        for key, value in initial_state.items():
            self._episode.add(f"initial_state/{key}", value.detach().clone())

    def add_step(
        self,
        obs_dict: dict[str, torch.Tensor],
        actions: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        state_dict: dict[str, torch.Tensor] | None = None,
    ):
        self._episode.add("actions", actions.detach().clone())

        for key, value in obs_dict.items():
            self._episode.add(f"obs/{key}", value.detach().clone())

        if state_dict is not None:
            for key, value in state_dict.items():
                self._episode.add(f"states/{key}", value.detach().clone())

        self._episode.add("rewards", reward.detach().clone())
        self._episode.add("dones", done.detach().clone())

    def export_episode(self, success: bool) -> tuple[bool, Optional[str], int]:
        if self._episode.is_empty():
            return False, None, 0

        handler = self._dataset_handler
        demo_key = f"demo_{handler._demo_count}"

        self._episode.success = success
        self._prepare_episode_for_export()

        acts = self._episode.data.get("actions")
        if acts is None:
            num_steps = 0
        elif isinstance(acts, torch.Tensor):
            num_steps = int(acts.shape[0])
        else:
            num_steps = len(acts)

        handler.write_episode(self._episode)
        handler.flush()

        if success:
            self.exported_successful_episode_count += 1
        else:
            self.exported_failed_episode_count += 1

        self.reset_episode()
        return True, demo_key, num_steps

    def close(self):
        self._dataset_handler.close()
