#!/bin/bash
# Full yaml_handle probe: approach + contact + ArticuBot T_rel close + 600s hold.
cd ~/workspace/physvla_sim/collect

~/isaacsim/python.sh debug_link_contact_probe.py \
  --task_id close_laptop_lid \
  --livestream 2 \
  --mode yaml_handle_push \
  --probe_steps 400 \
  --hold_steps 180
