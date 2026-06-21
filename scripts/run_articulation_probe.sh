#!/bin/bash
# yaml_handle probe — 改 --mode 切换阶段：
#   yaml_handle_contact  只 reach handle（阶段 1）
#   yaml_handle_push     全链路 reach + T_rel close（阶段 2）
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id close_the_microwave \
  --livestream 2 \
  --mode yaml_handle_push \
  --episode_step_limit 350 \
  --repeat \
  --max-attempts 10 \
  --repeat-delay 1.0
