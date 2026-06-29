#!/bin/bash
# Reach yaml handle contact only — print diagnostic report and hold scene open.
# 只接触到handle点就停下来。
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id lower_the_lamp \
  --livestream 2 \
  --mode yaml_handle_contact \
  --episode_step_limit 800 \
  --debug-logs
