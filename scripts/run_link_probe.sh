#!/bin/bash
# Phase C: visualize mesh contacts + move arm to top-ranked approach point.
# 旧 mesh 接触点排序，Phase C 调试用
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id close_laptop_lid \
  $ISAAC_LAB_STREAMING_ARGS \
  --mode top_contact \
  --probe_steps 400 \
  --hold_steps 120
