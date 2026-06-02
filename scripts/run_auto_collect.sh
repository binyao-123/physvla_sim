#!/bin/bash
# Auto collect: yaml_handle push + hinge-relative close
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh auto_trajectory_collection.py \
  --task_id close_laptop_lid \
  --headless \
  --num_demos 10000 \
  --max_attempts 20000 \
  --episode_step_limit 600 \
  --seed 42
