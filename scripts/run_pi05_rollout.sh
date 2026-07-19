#!/bin/bash
set -euo pipefail

cd ~/workspace/physvla_sim/rollout_sim
# 仿真推理启动脚本


# === 任务：调节显示器，提示词：adjust the display. ===
# ~/isaacsim/python.sh pi05_policy_rollout.py \
#   --task_id adjust_the_monitor \
#   --headless \
#   --num_episodes 2 \
#   --max_steps 600 \
#   --policy.path /home/ubuntu/workspace/lerobot/outputs/pi05_adjust_the_monitor_sim_15k_v1/checkpoints/last/pretrained_model \
#   --output_dir /home/ubuntu/workspace/physvla_sim/rollout_sim/rollouts/adjust_the_monitor_$(date +%Y%m%d_%H%M%S) \
#   --video_layout head_wrist \
#   --video_fps 10

# === 任务：关闭笔记本，提示词：close the laptop lid. ===
POLICY_PATH="${POLICY_PATH:-/home/ubuntu/workspace/lerobot/outputs/pi05_close_laptop_lid_sim_20k_v2/checkpoints/020000/pretrained_model}"
NUM_EPISODES="${NUM_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-600}"
REPLAN_HZ="${REPLAN_HZ:-3}"
MAX_JOINT_SPEED_RAD_S="${MAX_JOINT_SPEED_RAD_S:-0.9}"
RANDOMIZATION_MODE="${RANDOMIZATION_MODE:-all}"

case "$RANDOMIZATION_MODE" in
  all)
    TASK_POSE_ARGS=()
    RANDOMIZATION_ARGS=(
      --rand_camera_main
      --rand_lighting
      --rand_environment_scene
      --rand_clutter
    )
    ;;
  off)
    TASK_POSE_ARGS=(--fixed_task_pose)
    RANDOMIZATION_ARGS=(
      --no-rand_camera_main
      --no-rand_lighting
      --no-rand_environment_scene
      --no-rand_clutter
    )
    ;;
  *)
    echo "[ERROR] RANDOMIZATION_MODE must be 'all' or 'off', got: $RANDOMIZATION_MODE" >&2
    exit 2
    ;;
esac

# 训练产物可能携带当前 Isaac 推理端 PI05Config 已移除的字段。
# 推理前根据本机运行时 schema 过滤这些字段。
echo "[INFO] POLICY_PATH=$POLICY_PATH"
echo "[INFO] RANDOMIZATION_MODE=$RANDOMIZATION_MODE"
~/isaacsim/python.sh align_pi05_config.py \
  --policy-path "$POLICY_PATH"

~/isaacsim/python.sh pi05_policy_rollout.py \
  --task_id close_laptop_lid \
  "${TASK_POSE_ARGS[@]}" \
  --headless \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --realtime_chunking \
  --replan_hz "$REPLAN_HZ" \
  --ensemble_k 0.0625 \
  --max_joint_speed_rad_s "$MAX_JOINT_SPEED_RAD_S" \
  --policy.path "$POLICY_PATH" \
  --output_dir /home/ubuntu/workspace/physvla_sim/rollout_sim/rollouts/close_laptop_lid_$(date +%Y%m%d_%H%M%S) \
  --video_layout head_wrist \
  --video_fps 10 \
  "${RANDOMIZATION_ARGS[@]}"
