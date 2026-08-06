#!/bin/bash
set -euo pipefail
cd ~/workspace/lerobot

# --- Hub / tokenizer offline (same intent as rollout) ---
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# Keep proxy for optional non-Hub traffic; Hub stays offline above.
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"

POLICY_PRETRAINED_SRC="${POLICY_PRETRAINED_SRC:-/home/ubuntu/.cache/openpi/checkpoints/pi05_base}"
# Writable shadow of base: symlink weights, real file for preprocessor so align can
# rewrite tokenizer_name without needing write access to the shared cache.
POLICY_PRETRAINED="${POLICY_PRETRAINED:-$HOME/workspace/lerobot/outputs/_pi05_base_aligned}"

echo "[INFO] HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "[INFO] base_src=$POLICY_PRETRAINED_SRC"
echo "[INFO] base_aligned=$POLICY_PRETRAINED"

mkdir -p "$POLICY_PRETRAINED"
for f in "$POLICY_PRETRAINED_SRC"/*; do
  name="$(basename "$f")"
  if [[ "$name" == "policy_preprocessor.json" ]]; then
    cp -f "$f" "$POLICY_PRETRAINED/$name"
    continue
  fi
  if [[ -e "$POLICY_PRETRAINED/$name" || -L "$POLICY_PRETRAINED/$name" ]]; then
    continue
  fi
  ln -s "$f" "$POLICY_PRETRAINED/$name"
done

uv run python ~/workspace/physvla_sim/rollout_sim/align_pi05_config.py \
  --policy-path "$POLICY_PRETRAINED"

# === 任务：放下台灯（no_random_r2 / 300 ep），轻量微调 5k steps ===
# 配方：bf16 + gradient_checkpointing + train_expert_only；每 5k 存 ckpt；不写 training_state
uv run lerobot-train \
  --dataset.repo_id=physvla/piper_dual14_lower_the_lamp_no_random_r2 \
  --dataset.root=/home/ubuntu/workspace/physvla_sim/convert_data/lerobot_datasets/piper_dual14_lower_the_lamp_no_random_r2 \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path="$POLICY_PRETRAINED" \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=false \
  --policy.device=cuda \
  --batch_size=10 \
  --num_workers=4 \
  --prefetch_factor=4 \
  --steps=5000 \
  --save_freq=5000 \
  --save_training_state=false \
  --log_freq=50 \
  --wandb.enable=false \
  --output_dir=./outputs/pi05_lower_the_lamp_sim_5k_no_random_r2_v1 \
  --job_name=pi05_lower_the_lamp_sim_5k_no_random_r2_v1 \
  --overwrite_output_dir=false
