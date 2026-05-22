#!/bin/bash
cd ~/workspace/physvla_sim/sim_rollout
# 仿真推理启动脚本


# === 任务：调节显示器，提示词：adjust the display. ===
# ~/isaacsim/python.sh pi05_policy_rollout.py \
#   --task_id adjust_the_monitor \
#   --headless \
#   --num_episodes 2 \
#   --max_steps 600 \
#   --policy.path /home/ubuntu/workspace/lerobot/outputs/pi05_adjust_the_monitor_sim_15k_v1/checkpoints/last/pretrained_model \
#   --output_dir /home/ubuntu/workspace/physvla_sim/sim_rollout/rollouts/adjust_the_monitor_$(date +%Y%m%d_%H%M%S) \
#   --video_layout head_wrist \
#   --video_fps 10

# === 任务：关闭笔记本，提示词：Close the laptop lid until it is fully closed. ===
~/isaacsim/python.sh pi05_policy_rollout.py \
  --task_id close_laptop_lid \
  --headless \
  --num_episodes 2 \
  --max_steps 600 \
  --policy.path /home/ubuntu/workspace/lerobot/outputs/pi05_close_laptop_lid_sim_5k_v1/checkpoints/last/pretrained_model \
  --output_dir /home/ubuntu/workspace/physvla_sim/sim_rollout/rollouts/close_laptop_lid_$(date +%Y%m%d_%H%M%S) \
  --video_layout head_wrist \
  --video_fps 10
