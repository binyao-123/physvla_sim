#!/bin/bash
set -euo pipefail

cd ~/workspace/physvla_sim/rollout_sim
# PLD 第二阶段数据采集：基座策略探测 T_probe 步 → 脚本化专家从漂移状态接管把任务做完。
# 整条轨迹（含 probing 前缀）一起存，probing 段就是专家要学会恢复的那个分布。
#
# 与 run_prts_rollout.sh 的区别：这里不是评测，是采数据。场景初始化 / DR / 种子派生
# 完全一致，所以同一个 SEED 下 PLD episode 与 BC 评测 episode 的初始场景是同一个。

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi cannot talk to the kernel driver; fix the GPU driver first." >&2
  exit 1
fi

TASK_ID="${TASK_ID:-close_laptop_lid}"
POLICY_SERVER="${POLICY_SERVER:-127.0.0.1:10093}"
PRTS_ROOT="${PRTS_ROOT:-/home/ubuntu/workspace/PRTS}"
NUM_EPISODES="${NUM_EPISODES:-5}"
# 蒸馏只用成功的轨迹，所以放量时按"采到多少条成功"算，而不是跑固定次数。
# 设为 0 则回到 NUM_EPISODES 的固定次数行为。
TARGET_SUCCESSES="${TARGET_SUCCESSES:-0}"
# 0 = 不设尝试上限，一直采到 TARGET_SUCCESSES 条成功为止。
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
# 补采时必须设置：场景由 (SEED, episode_index) 决定，从 0 重开会重复采到同样的场景。
EPISODE_INDEX_OFFSET="${EPISODE_INDEX_OFFSET:-0}"
MAX_STEPS="${MAX_STEPS:-600}"
MAX_JOINT_SPEED_RAD_S="${MAX_JOINT_SPEED_RAD_S:-0.9}"
ACTION_DIM="${ACTION_DIM:-7}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-50}"
STATE_RELATIVE_ACTION="${STATE_RELATIVE_ACTION:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
RANDOMIZATION_MODE="${RANDOMIZATION_MODE:-all}"
# PLD: T_probe ~ U[PROBE_MIN_STEPS, PROBE_ALPHA*MAX_STEPS]，由 (SEED, episode) 派生。
PROBE_ALPHA="${PROBE_ALPHA:-0.2}"
PROBE_MIN_STEPS="${PROBE_MIN_STEPS:-10}"
SEED="${SEED:-}"
REPLAY_EPISODE_CONFIG="${REPLAY_EPISODE_CONFIG:-}"

if [[ -z "$POLICY_SERVER" ]]; then
  echo "[ERROR] PLD collection needs the deployed base policy; set POLICY_SERVER=host:port." >&2
  exit 2
fi
if [[ ! -d "$PRTS_ROOT/third_party/openpi-client/src/openpi_client" ]]; then
  echo "[ERROR] openpi_client missing at $PRTS_ROOT/third_party/openpi-client/src" >&2
  exit 2
fi
if [[ ! -f "../collect/task_configs/${TASK_ID}.yaml" ]]; then
  echo "[ERROR] no expert interaction config: collect/task_configs/${TASK_ID}.yaml" >&2
  exit 2
fi

case "$RANDOMIZATION_MODE" in
  all)
    TASK_POSE_ARGS=()
    RANDOMIZATION_ARGS=(--rand_camera_main --rand_lighting --rand_environment_scene --rand_clutter)
    ;;
  off)
    TASK_POSE_ARGS=(--fixed_task_pose)
    RANDOMIZATION_ARGS=(--no-rand_camera_main --no-rand_lighting --no-rand_environment_scene --no-rand_clutter)
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

case "$SAVE_VIDEO" in
  1|true|TRUE|yes|YES) VIDEO_ARGS=(--save_video) ;;
  0|false|FALSE|no|NO) VIDEO_ARGS=(--no-save_video) ;;
  *)
    echo "[ERROR] SAVE_VIDEO must be 0 or 1, got: $SAVE_VIDEO" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/workspace/physvla_sim/rollout_sim/rollouts/PRTS_PLD/${TASK_ID}_$(date +%Y%m%d_%H%M%S)}"

echo "[INFO] TASK_ID=$TASK_ID"
echo "[INFO] POLICY_SERVER=$POLICY_SERVER (base policy = probing segment)"
echo "[INFO] EXPERT=PushInteractionExecutor.takeover (collect/task_configs/${TASK_ID}.yaml)"
if [[ "$TARGET_SUCCESSES" -gt 0 ]]; then
  if [[ "$MAX_ATTEMPTS" -gt 0 ]]; then
    echo "[INFO] TARGET_SUCCESSES=$TARGET_SUCCESSES MAX_ATTEMPTS=$MAX_ATTEMPTS MAX_STEPS=$MAX_STEPS"
  else
    echo "[INFO] TARGET_SUCCESSES=$TARGET_SUCCESSES MAX_ATTEMPTS=<none> MAX_STEPS=$MAX_STEPS"
  fi
else
  echo "[INFO] NUM_EPISODES=$NUM_EPISODES MAX_STEPS=$MAX_STEPS"
fi
echo "[INFO] PROBE_ALPHA=$PROBE_ALPHA PROBE_MIN_STEPS=$PROBE_MIN_STEPS"
echo "[INFO] ACTION_EXEC_HORIZON=$ACTION_EXEC_HORIZON MAX_JOINT_SPEED_RAD_S=$MAX_JOINT_SPEED_RAD_S"
echo "[INFO] RANDOMIZATION_MODE=$RANDOMIZATION_MODE SEED=${SEED:-<auto>}"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"

~/isaacsim/python.sh prts_pld_rollout.py \
  --task_id "$TASK_ID" \
  "${TASK_POSE_ARGS[@]}" \
  --headless \
  --num_episodes "$NUM_EPISODES" \
  --target_successes "$TARGET_SUCCESSES" \
  --max_attempts "$MAX_ATTEMPTS" \
  --episode_index_offset "$EPISODE_INDEX_OFFSET" \
  --max_steps "$MAX_STEPS" \
  --max_joint_speed_rad_s "$MAX_JOINT_SPEED_RAD_S" \
  --policy_server "$POLICY_SERVER" \
  --prts_root "$PRTS_ROOT" \
  --action_dim "$ACTION_DIM" \
  --action_exec_horizon "$ACTION_EXEC_HORIZON" \
  "${REL_ARGS[@]}" \
  --probe_alpha "$PROBE_ALPHA" \
  --probe_min_steps "$PROBE_MIN_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  "${VIDEO_ARGS[@]}" \
  --video_layout head_wrist \
  --video_fps 10 \
  "${SEED_ARGS[@]}" \
  "${REPLAY_ARGS[@]}" \
  "${RANDOMIZATION_ARGS[@]}"

# 采集的是"专家做成了的"轨迹，所以这里看的是接管成功率，不是基座策略的 NJC。
# 清掉 PYTHONPATH：isaacsim 的 python.sh 会把它指向 kit 自带的 3.11 stdlib，
# 系统 python3 加载那套 stdlib 会在 re 模块上炸 "SRE module mismatch"。
env -u PYTHONPATH python3 eval_njc.py "$OUTPUT_DIR"
