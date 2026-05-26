#!/bin/bash
# Auto collect: ArticuBot mesh sampling + hinge-relative push (Phase C)
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh auto_trajectory_collection.py \
  --task_id close_laptop_lid \
  --headless \
  --num_demos 1 \
  --max_attempts 3 \
  --seed 42 \
  --save_failed \
  --no-debug_hardcoded_push
