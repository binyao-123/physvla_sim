#!/bin/bash
cd ~/workspace/lerobot

# --- 连接Hugging Face Hub---
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"

# === 任务：调节显示器，提示词：adjust the display. ===
# uv run lerobot-train \
#   --dataset.repo_id=physvla/adjust_the_monitor_sim_r1 \
#   --dataset.root=/home/ubuntu/workspace/physvla_sim/lerobot_datasets/adjust_the_monitor_sim_r1 \
#   --dataset.video_backend=pyav \
#   --policy.type=pi05 \
#   --policy.pretrained_path=/home/ubuntu/.cache/openpi/checkpoints/pi05_base \
#   --policy.push_to_hub=false \
#   --policy.dtype=bfloat16 \
#   --policy.gradient_checkpointing=true \
#   --policy.train_expert_only=true \
#   --policy.freeze_vision_encoder=false \
#   --policy.device=cuda \
#   --batch_size=10 \
#   --num_workers=4 \
#   --prefetch_factor=4 \
#   --steps=15000 \
#   --save_freq=5000 \
#   --log_freq=50 \
#   --wandb.enable=true \
#   --wandb.project=physvla-pi05 \
#   --wandb.disable_artifact=true \
#   --output_dir=./outputs/pi05_adjust_the_monitor_sim_15k_v1 \
#   --job_name=pi05_adjust_the_monitor_sim_15k_v1 \
#   --overwrite_output_dir=false


# === 任务：关闭笔记本，提示词：Close the laptop lid until it is fully closed. ===
uv run lerobot-train \
  --dataset.repo_id=physvla/piper_dual14_close_laptop_lid_sim_r1 \
  --dataset.root=/home/ubuntu/workspace/physvla_sim/lerobot_datasets/piper_dual14_close_laptop_lid_sim_r1 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path=/home/ubuntu/.cache/openpi/checkpoints/pi05_base \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=false \
  --policy.device=cuda \
  --batch_size=10 \
  --num_workers=4 \
  --prefetch_factor=4 \
  --steps=5000 \
  --save_freq=5000 \
  --log_freq=50 \
  --wandb.enable=true \
  --wandb.project=physvla-pi05 \
  --wandb.disable_artifact=true \
  --output_dir=./outputs/pi05_close_laptop_lid_sim_5k_v1 \
  --job_name=pi05_close_laptop_lid_sim_5k_v1 \
  --overwrite_output_dir=true

