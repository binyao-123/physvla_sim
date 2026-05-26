# Reference algorithms (ArticuBot-derived, no ArticuBot runtime dependency)

| File | Source | Notes |
|------|--------|-------|
| `grasping_utils.py` | `ArticuBot/manipulation/grasping_utils.py` | Gripper z-align with surface normal; numpy/scipy only |
| `opening_kinematics.py` | `ArticuBot/manipulation/gpt_primitive_api.py` `open_door()` | Relative EE pose under link articulation |

Adapted for Isaac Lab + Piper; no PyBullet / open3d imports.
