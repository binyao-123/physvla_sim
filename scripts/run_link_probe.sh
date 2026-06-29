#!/bin/bash
# Phase C: visualize mesh contacts + move arm to top-ranked approach point.
# 旧 mesh 接触点排序，Phase C 调试用
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id lower_the_lamp \
  --livestream 2 \
  --mode top_contact \
  --episode_step_limit 800 \
  --hold_steps 120
