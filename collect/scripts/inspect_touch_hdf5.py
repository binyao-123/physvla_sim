#!/usr/bin/env python3
"""Offline touch calibration: peak EE frame → link-local handle for task yaml.

This is the ONLY allowed use of reference HDF5 for auto collection.
Copy printed contact_pos_link / contact_quat_wxyz_link into task_configs/*.yaml.
Does not run Isaac Sim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_COLLECT = Path(__file__).resolve().parents[1]
if str(_COLLECT) not in sys.path:
    sys.path.insert(0, str(_COLLECT))

from reference.contact_reference import load_touch_contact_from_hdf5, summarize_touch_reference

# scene.usd default layout (no domain randomization) — same as probe logs.
DEFAULT_LINK_POS = [0.3683, -0.0194, 0.0007]
DEFAULT_LINK_QUAT_WXYZ = (0.5065, 0.0729, 0.0729, 0.0869)
DEFAULT_HINGE_ORIGIN = [0.3683, -0.0194, 0.0007]
DEFAULT_HINGE_AXIS = [1.0, 0.0, 0.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect touch contact from keyboard HDF5.")
    parser.add_argument("hdf5", type=Path, help="Path to keyboard HDF5")
    parser.add_argument("--demo", type=int, default=0)
    args = parser.parse_args()

    import numpy as np

    ref = load_touch_contact_from_hdf5(
        args.hdf5.expanduser().resolve(),
        args.demo,
        link_pos_world=np.asarray(DEFAULT_LINK_POS, dtype=np.float64),
        link_quat_wxyz=DEFAULT_LINK_QUAT_WXYZ,
        hinge_origin_world=np.asarray(DEFAULT_HINGE_ORIGIN, dtype=np.float64),
        hinge_axis_world=np.asarray(DEFAULT_HINGE_AXIS, dtype=np.float64),
    )
    print(summarize_touch_reference(ref))
    print("YAML snippet (link-local handle for close_laptop_lid.yaml):")
    print(f"  push_contact_offset_link: {np.round(ref.contact_pos_link, 4).tolist()}")
    print(f"  contact_quat_link: {np.round(ref.contact_quat_wxyz_link, 4).tolist()}")


if __name__ == "__main__":
    main()
