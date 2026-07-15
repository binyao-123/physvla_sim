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
POLICY_PATH=/home/ubuntu/workspace/lerobot/outputs/pi05_close_laptop_lid_sim_5k_v1/checkpoints/last/pretrained_model

# 训练产物可能携带当前 Isaac 推理端 PI05Config 已移除的字段。
# 推理前根据本机运行时 schema 过滤这些字段。
~/isaacsim/python.sh align_pi05_config.py --policy-path "$POLICY_PATH" --normalization-mode QUANTILES

~/isaacsim/python.sh pi05_policy_rollout.py \
  --task_id close_laptop_lid \
  --headless \
  --num_episodes 2 \
  --max_steps 600 \
  --policy.path "$POLICY_PATH" \
  --output_dir /home/ubuntu/workspace/physvla_sim/rollout_sim/rollouts/close_laptop_lid_$(date +%Y%m%d_%H%M%S) \
  --video_layout head_wrist \
  --video_fps 10 \
  --rand_camera_main \
  --rand_lighting \
  --rand_environment_scene \
  --rand_clutter
