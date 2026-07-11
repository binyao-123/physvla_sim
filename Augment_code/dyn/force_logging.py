"""CSV logging for simulated Piper gripper contact forces."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


class ForceCsvLogger:
    """Write one force time-series CSV and metadata sidecar per demo slot."""

    _FIELDNAMES = (
        "sim_time_sec",
        "wall_time_sec",
        "demo_slot",
        "left_fx_n",
        "left_fy_n",
        "left_fz_n",
        "left_force_norm_n",
        "right_fx_n",
        "right_fy_n",
        "right_fz_n",
        "right_force_norm_n",
        "total_fx_n",
        "total_fy_n",
        "total_fz_n",
        "total_force_norm_n",
        "in_contact",
    )

    def __init__(
        self,
        output_dir: Path,
        *,
        task_id: str,
        contact_prim_paths: Sequence[str],
    ) -> None:
        self.output_dir = output_dir
        self.task_id = task_id
        self.contact_prim_paths = list(contact_prim_paths)
        self._file = None
        self._writer = None
        self._path: Path | None = None
        self._demo_slot: int | None = None
        self._row_count = 0

    @property
    def path(self) -> Path | None:
        return self._path

    def start(self, demo_slot: int) -> Path:
        if self._file is not None:
            raise RuntimeError("Force CSV is already open. Close it before starting another demo.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self._path = self.output_dir / f"{self.task_id}_{timestamp}_slot{demo_slot:03d}.csv"
        self._demo_slot = demo_slot
        self._row_count = 0
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        return self._path

    def log(
        self,
        *,
        sim_time_sec: float,
        wall_time_sec: float,
        left_force_n: Sequence[float],
        right_force_n: Sequence[float],
        total_force_n: Sequence[float],
        in_contact: bool,
    ) -> None:
        if self._writer is None or self._file is None or self._demo_slot is None:
            raise RuntimeError("Start a force CSV before logging force samples.")

        def _norm(force: Sequence[float]) -> float:
            return sum(float(component) ** 2 for component in force) ** 0.5

        self._writer.writerow(
            {
                "sim_time_sec": f"{sim_time_sec:.6f}",
                "wall_time_sec": f"{wall_time_sec:.6f}",
                "demo_slot": self._demo_slot,
                "left_fx_n": f"{float(left_force_n[0]):.6f}",
                "left_fy_n": f"{float(left_force_n[1]):.6f}",
                "left_fz_n": f"{float(left_force_n[2]):.6f}",
                "left_force_norm_n": f"{_norm(left_force_n):.6f}",
                "right_fx_n": f"{float(right_force_n[0]):.6f}",
                "right_fy_n": f"{float(right_force_n[1]):.6f}",
                "right_fz_n": f"{float(right_force_n[2]):.6f}",
                "right_force_norm_n": f"{_norm(right_force_n):.6f}",
                "total_fx_n": f"{float(total_force_n[0]):.6f}",
                "total_fy_n": f"{float(total_force_n[1]):.6f}",
                "total_fz_n": f"{float(total_force_n[2]):.6f}",
                "total_force_norm_n": f"{_norm(total_force_n):.6f}",
                "in_contact": int(in_contact),
            }
        )
        self._file.flush()
        self._row_count += 1

    def close(self, *, outcome: str) -> Path | None:
        if self._file is None or self._path is None:
            return None

        self._file.close()
        metadata_path = self._path.with_suffix(".json")
        metadata: Mapping[str, object] = {
            "task_id": self.task_id,
            "demo_slot": self._demo_slot,
            "outcome": outcome,
            "sample_count": self._row_count,
            "contact_prim_paths": self.contact_prim_paths,
            "force_frame": "world",
            "force_unit": "N",
            "csv_file": self._path.name,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        closed_path = self._path
        self._file = None
        self._writer = None
        self._path = None
        self._demo_slot = None
        return closed_path
