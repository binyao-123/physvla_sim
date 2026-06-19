#!/bin/bash
# yaml_handle_push: approach + contact + T_rel close push
# 成功条件: joint_1 > 0° (task_registry rollout_success_specs)
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id adjust_the_monitor \
  --livestream 2 \
  --mode yaml_handle_push \
  --probe_steps 400 \
  --episode_step_limit 800 \
  --hold_steps 180 \
  --trace-ee-handle \
  --trace-interval 30 \
  --repeat \
  --max-attempts 10 \
  --repeat-delay 1.0 \
  --debug-logs
