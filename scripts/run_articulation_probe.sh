#!/bin/bash
# yaml_handle_push: approach + world -Z push until joint_0 < -25°
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id lower_the_lamp \
  --livestream 2 \
  --mode yaml_handle_push \
  --episode_step_limit 700 \
  --hold_steps 180 \
  --trace-ee-handle \
  --trace-interval 30 \
  --repeat \
  --max-attempts 10 \
  --repeat-delay 1.0 \
  --debug-logs
