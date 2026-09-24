#!/bin/bash
set -euo pipefail

cd ~/workspace/physvla_sim/rollout_sim
# PRTS 仿真推理 / 评测启动脚本
# 转轴完成进度：0% = 相对初值无变化，100% = 达到 success 目标角；对 NUM_EPISODES 个 episode 取均值。

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

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
# 当前本地权重：piper left6 adjust_the_monitor 30k SFT
#   cameras: head + left_wrist only
#   state/action: 7-D left arm (6 joints + gripper); gripper always 0
#   chunk=50 → ACTION_EXEC_HORIZON=50
# 关闭笔记本：TASK_ID=close_laptop_lid
# POLICY_PATH=.../prts_piper_left6_close_laptop_lid_ep1k_ck50_30k/checkpoint-final-30000
#
# 远程模式：POLICY_SERVER=127.0.0.1:10093（走 A100 上的 websocket policy server）
#   此时不读本地 POLICY_PATH，服务端已做反归一化 + state-relative 反变换，
#   客户端拿到的就是绝对关节角。TASK_ID 必须与服务端当前 serve 的任务一致。
TASK_ID="${TASK_ID:-adjust_the_monitor}"
POLICY_SERVER="${POLICY_SERVER:-}"
POLICY_PATH="${POLICY_PATH:-/home/ubuntu/Downloads/prts_piper_left6_adjust_the_monitor_ep1k_ck50_30k/checkpoint-final-30000}"
PRTS_ROOT="${PRTS_ROOT:-/home/ubuntu/workspace/PRTS}"
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
MAX_JOINT_SPEED_RAD_S="${MAX_JOINT_SPEED_RAD_S:-0.9}"
ACTION_DIM="${ACTION_DIM:-7}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-50}"
STATE_RELATIVE_ACTION="${STATE_RELATIVE_ACTION:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
RANDOMIZATION_MODE="${RANDOMIZATION_MODE:-all}"
# SEED 固定后，第 N 条 episode 的场景由 (SEED, N) 唯一决定，与前面跑过几条无关。
# 留空则自动生成一个并记进 run_meta.json 的 run_seed，事后照样可复现。
SEED="${SEED:-}"
# REPLAY_EPISODE_CONFIG 指向某次 rollout 的输出目录 / episode_configs 目录 / 单个
# episode_XXXX.json，此时不再采样，逐条复现保存下来的初始配置。
REPLAY_EPISODE_CONFIG="${REPLAY_EPISODE_CONFIG:-}"

if [[ -n "$POLICY_SERVER" ]]; then
  POLICY_ARGS=(--policy_server "$POLICY_SERVER")
  if [[ ! -d "$PRTS_ROOT/third_party/openpi-client/src/openpi_client" ]]; then
    echo "[ERROR] openpi_client missing at $PRTS_ROOT/third_party/openpi-client/src" >&2
    exit 2
  fi
else
  POLICY_ARGS=(--policy.path "$POLICY_PATH")
  if [[ ! -f "$POLICY_PATH/config.json" ]]; then
    echo "[ERROR] POLICY_PATH is not a PRTS checkpoint (missing config.json): $POLICY_PATH" >&2
    exit 2
  fi
  if [[ ! -f "$POLICY_PATH/statistics.json" ]]; then
    echo "[ERROR] POLICY_PATH missing statistics.json: $POLICY_PATH" >&2
    exit 2
  fi
fi
if [[ ! -d "$PRTS_ROOT/prts" ]]; then
  echo "[ERROR] PRTS repo missing at $PRTS_ROOT (expected clone of https://github.com/TeleHuman/PRTS)" >&2
  exit 2
fi

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

SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=(--seed "$SEED")
fi

REPLAY_ARGS=()
if [[ -n "$REPLAY_EPISODE_CONFIG" ]]; then
  if [[ ! -e "$REPLAY_EPISODE_CONFIG" ]]; then
    echo "[ERROR] REPLAY_EPISODE_CONFIG not found: $REPLAY_EPISODE_CONFIG" >&2
    exit 2
  fi
  REPLAY_ARGS=(--replay_episode_config "$REPLAY_EPISODE_CONFIG")
fi

case "$STATE_RELATIVE_ACTION" in
  1|true|TRUE|yes|YES) REL_ARGS=(--state_relative_action) ;;
  0|false|FALSE|no|NO) REL_ARGS=(--no-state_relative_action) ;;
  *)
    echo "[ERROR] STATE_RELATIVE_ACTION must be 0 or 1, got: $STATE_RELATIVE_ACTION" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/workspace/physvla_sim/rollout_sim/rollouts/PRTS/${TASK_ID}_$(date +%Y%m%d_%H%M%S)}"

echo "[INFO] TASK_ID=$TASK_ID"
if [[ -n "$POLICY_SERVER" ]]; then
  echo "[INFO] POLICY_SERVER=$POLICY_SERVER (remote, local ckpt unused)"
else
  echo "[INFO] POLICY_PATH=$POLICY_PATH"
fi
echo "[INFO] PRTS_ROOT=$PRTS_ROOT"
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
echo "[INFO] ACTION_DIM=$ACTION_DIM"
echo "[INFO] ACTION_EXEC_HORIZON=$ACTION_EXEC_HORIZON"
echo "[INFO] STATE_RELATIVE_ACTION=$STATE_RELATIVE_ACTION"
echo "[INFO] SAVE_VIDEO=$SAVE_VIDEO"
echo "[INFO] RANDOMIZATION_MODE=$RANDOMIZATION_MODE"
echo "[INFO] SEED=${SEED:-<auto>}"
echo "[INFO] REPLAY_EPISODE_CONFIG=${REPLAY_EPISODE_CONFIG:-<none>}"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] control_path=prts_sample_actions+slew"
echo "[INFO] HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

~/isaacsim/python.sh prts_policy_rollout.py \
  --task_id "$TASK_ID" \
  "${TASK_POSE_ARGS[@]}" \
  --headless \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --max_joint_speed_rad_s "$MAX_JOINT_SPEED_RAD_S" \
  "${POLICY_ARGS[@]}" \
  --prts_root "$PRTS_ROOT" \
  --action_dim "$ACTION_DIM" \
  --action_exec_horizon "$ACTION_EXEC_HORIZON" \
  "${REL_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  "${VIDEO_ARGS[@]}" \
  --video_layout head_wrist \
  --video_fps 10 \
  "${SEED_ARGS[@]}" \
  "${REPLAY_ARGS[@]}" \
  "${RANDOMIZATION_ARGS[@]}"

# 主指标 NJC 用终端关节角（FITR 5.1），峰值仅作诊断。cwd 已在脚本开头切到 rollout_sim。
python3 eval_njc.py "$OUTPUT_DIR"
