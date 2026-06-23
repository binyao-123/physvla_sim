#!/bin/bash
# yaml_handle_push: approach + contact + prismatic close (full pipeline)
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id close_the_drawer \
  --livestream 2 \
  --mode yaml_handle_push \
  --episode_step_limit 250 \
  --no-trace-ee-handle \
  --repeat \
  --max-attempts 10 \
  --repeat-delay 1.0
