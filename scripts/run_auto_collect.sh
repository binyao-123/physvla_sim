#!/bin/bash
# Auto collect: yaml_handle push + hinge-relative close
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh auto_trajectory_collection.py \
  --task_id close_laptop_lid \
  --headless \
  --num_demos 5 \
  --seed 42 \
  --save_failed
