#!/bin/bash
set -euo pipefail

cd ~/workspace/physvla_sim/rollout_sim
# 仿真推理 / 通用评测启动脚本（纯净 select_action）
# 转轴完成进度：0% = 相对初值无变化，100% = 达到 success 目标角；对 NUM_EPISODES 个 episode 取均值。

# Prefer local HF cache for tokenizer / hub assets (avoid SSL flakes via proxy).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# GPU / NVIDIA driver preflight — catch half-upgraded drivers before Isaac starts.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. Install/repair NVIDIA driver first." >&2
  exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi cannot talk to the kernel driver." >&2
  echo "[ERROR] Typical cause: userspace driver upgraded but kernel module not reloaded." >&2
  echo "[ERROR] Check: cat /sys/module/nvidia/version   vs   ls -l /usr/lib/x86_64-linux-gnu/libcuda.so.1" >&2
  echo "[ERROR] Fix: sudo reboot  (or finish dpkg configure, then reboot)." >&2
  nvidia-smi 2>&1 | sed 's/^/[ERROR] /' || true
  exit 1
fi
echo "[INFO] GPU preflight ok:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/[INFO]   /'


# === 切换任务：改 TASK_ID + POLICY_PATH（或环境变量覆盖）===
# 调节显示器：TASK_ID=adjust_the_monitor
# POLICY_PATH=.../pi05_adjust_the_monitor_sim_15k_v1/checkpoints/last/pretrained_model
# 关闭笔记本：
TASK_ID="${TASK_ID:-close_laptop_lid}"
POLICY_PATH="${POLICY_PATH:-/home/ubuntu/workspace/lerobot/outputs/pi05_close_laptop_lid_sim_80k_v3/checkpoints/080000/pretrained_model}"
NUM_EPISODES="${NUM_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-600}"
MAX_JOINT_SPEED_RAD_S="${MAX_JOINT_SPEED_RAD_S:-0.9}"
# 1=保存预览 MP4，0=不保存（大批量评测推荐关）
SAVE_VIDEO="${SAVE_VIDEO:-0}"
RANDOMIZATION_MODE="${RANDOMIZATION_MODE:-all}"

case "$RANDOMIZATION_MODE" in
  all)
    TASK_POSE_ARGS=()
    RANDOMIZATION_ARGS=(
      --rand_camera_main
      --rand_lighting
      --rand_environment_scene
      --rand_clutter
    )
    ;;
  off)
    TASK_POSE_ARGS=(--fixed_task_pose)
    RANDOMIZATION_ARGS=(
      --no-rand_camera_main
      --no-rand_lighting
      --no-rand_environment_scene
      --no-rand_clutter
    )
    ;;
  *)
    echo "[ERROR] RANDOMIZATION_MODE must be 'all' or 'off', got: $RANDOMIZATION_MODE" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/workspace/physvla_sim/rollout_sim/rollouts/${TASK_ID}_$(date +%Y%m%d_%H%M%S)}"

echo "[INFO] TASK_ID=$TASK_ID"
echo "[INFO] POLICY_PATH=$POLICY_PATH"
case "$SAVE_VIDEO" in
  1|true|TRUE|yes|YES) VIDEO_ARGS=(--save_video) ;;
  0|false|FALSE|no|NO) VIDEO_ARGS=(--no-save_video) ;;
  *)
    echo "[ERROR] SAVE_VIDEO must be 0 or 1, got: $SAVE_VIDEO" >&2
    exit 2
    ;;
esac

echo "[INFO] NUM_EPISODES=$NUM_EPISODES"
echo "[INFO] MAX_JOINT_SPEED_RAD_S=$MAX_JOINT_SPEED_RAD_S"
echo "[INFO] SAVE_VIDEO=$SAVE_VIDEO"
echo "[INFO] RANDOMIZATION_MODE=$RANDOMIZATION_MODE"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] control_path=select_action+slew"
echo "[INFO] HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

~/isaacsim/python.sh align_pi05_config.py \
  --policy-path "$POLICY_PATH"

~/isaacsim/python.sh pi05_policy_rollout.py \
  --task_id "$TASK_ID" \
  "${TASK_POSE_ARGS[@]}" \
  --headless \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --max_joint_speed_rad_s "$MAX_JOINT_SPEED_RAD_S" \
  --policy.path "$POLICY_PATH" \
  --output_dir "$OUTPUT_DIR" \
  "${VIDEO_ARGS[@]}" \
  --video_layout head_wrist \
  --video_fps 10 \
  "${RANDOMIZATION_ARGS[@]}"

# 主指标 NJC 用终端关节角（FITR 5.1），峰值仅作诊断。cwd 已在脚本开头切到 rollout_sim。
python3 eval_njc.py "$OUTPUT_DIR"
