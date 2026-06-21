#!/bin/bash
# Launch multiple isolated auto-collection watchdog workers on one host/GPU.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/workspace/physvla_sim}"
COLLECT_DIR="${COLLECT_DIR:-$ROOT_DIR/collect}"
TASK_ID="${TASK_ID:-close_the_microwave}"
NUM_WORKERS="${NUM_WORKERS:-3}"
TARGET_DEMOS_TOTAL="${TARGET_DEMOS_TOTAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-300}"
BASE_SEED="${BASE_SEED:-42}"
RESTART_SLEEP_SEC="${RESTART_SLEEP_SEC:-10}"
EPISODE_STEP_LIMIT="${EPISODE_STEP_LIMIT:-300}"
POST_RESET_WARMUP_SEC="${POST_RESET_WARMUP_SEC:-0.2}"
COLLECTOR_TIMEOUT_SEC="${COLLECTOR_TIMEOUT_SEC:-21600}"
STOP_AFTER_RUN_TARGET="${STOP_AFTER_RUN_TARGET:-1}"
STOP_AFTER_TARGET_GRACE_SEC="${STOP_AFTER_TARGET_GRACE_SEC:-20}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-5}"
FLAT_DATASET_SESSION_DIR="${FLAT_DATASET_SESSION_DIR:-1}"
PARALLEL_DATASET_ROOT="${PARALLEL_DATASET_ROOT:-$COLLECT_DIR/datasets/${TASK_ID}_parallel}"
PARALLEL_LOG_ROOT="${PARALLEL_LOG_ROOT:-$COLLECT_DIR/logs/watchdog_parallel}"
RUN_AUTO_COLLECT="${RUN_AUTO_COLLECT:-$ROOT_DIR/scripts/run_auto_collect.sh}"
START_STAGGER_SEC="${START_STAGGER_SEC:-20}"

if (( NUM_WORKERS < 1 )); then
  echo "NUM_WORKERS must be >= 1" >&2
  exit 2
fi

mkdir -p "$PARALLEL_DATASET_ROOT" "$PARALLEL_LOG_ROOT"

base_target=$((TARGET_DEMOS_TOTAL / NUM_WORKERS))
remainder=$((TARGET_DEMOS_TOTAL % NUM_WORKERS))

pids=()

echo "[PARALLEL] task=$TASK_ID workers=$NUM_WORKERS total_target=$TARGET_DEMOS_TOTAL batch_size=$BATCH_SIZE"
LANGUAGE_INSTRUCTION="$(python3 -c "import sys; sys.path.insert(0, '${COLLECT_DIR}'); from task_registry import get_task_preset; print(get_task_preset('${TASK_ID}').language_instruction)")"
echo "[PARALLEL] language_instruction=${LANGUAGE_INSTRUCTION}"
echo "[PARALLEL] dataset_root=$PARALLEL_DATASET_ROOT"
echo "[PARALLEL] log_root=$PARALLEL_LOG_ROOT"

for (( i=0; i<NUM_WORKERS; i++ )); do
  worker_id="w${i}"
  worker_target=$base_target
  if (( i < remainder )); then
    worker_target=$((worker_target + 1))
  fi
  worker_seed=$((BASE_SEED + i * 1009))
  worker_dataset_dir="$PARALLEL_DATASET_ROOT/$worker_id"
  worker_dataset_file="$worker_dataset_dir/${TASK_ID}_${worker_id}.hdf5"
  worker_log_dir="$PARALLEL_LOG_ROOT/$worker_id"
  mkdir -p "$worker_dataset_dir" "$worker_log_dir"

  echo "[PARALLEL] starting $worker_id target=$worker_target seed=$worker_seed dataset=$worker_dataset_file"
  (
    export COLLECT_DIR="$COLLECT_DIR"
    export TASK_ID="$TASK_ID"
    export WORKER_ID="$worker_id"
    export DATASET_DIR="$worker_dataset_dir"
    export DATASET_FILE="$worker_dataset_file"
    export DATASET_STEM="${TASK_ID}_${worker_id}"
    export TARGET_DEMOS="$worker_target"
    export BATCH_SIZE="$BATCH_SIZE"
    export MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"
    export EPISODE_STEP_LIMIT="$EPISODE_STEP_LIMIT"
    export SEED="$worker_seed"
    export RESTART_SLEEP_SEC="$RESTART_SLEEP_SEC"
    export POST_RESET_WARMUP_SEC="$POST_RESET_WARMUP_SEC"
    export COLLECTOR_TIMEOUT_SEC="$COLLECTOR_TIMEOUT_SEC"
    export STOP_AFTER_RUN_TARGET="$STOP_AFTER_RUN_TARGET"
    export STOP_AFTER_TARGET_GRACE_SEC="$STOP_AFTER_TARGET_GRACE_SEC"
    export WATCHDOG_POLL_SEC="$WATCHDOG_POLL_SEC"
    export FLAT_DATASET_SESSION_DIR="$FLAT_DATASET_SESSION_DIR"
    export LOG_DIR="$worker_log_dir"
    bash "$RUN_AUTO_COLLECT"
  ) > "$worker_log_dir/launcher.log" 2>&1 &
  pids+=("$!")

  if (( i < NUM_WORKERS - 1 && START_STAGGER_SEC > 0 )); then
    sleep "$START_STAGGER_SEC"
  fi
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status == 0 )); then
  echo "[PARALLEL] all workers completed successfully"
else
  echo "[PARALLEL] one or more workers failed; inspect $PARALLEL_LOG_ROOT" >&2
fi
exit "$status"
