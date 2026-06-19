#!/bin/bash
# adjust_the_monitor auto-collect with watchdog (yaml_handle push, VLM: "adjust the display.")
# Restarts after CUDA/render fatal errors or wrist-camera mount failures.
set -euo pipefail

COLLECT_DIR="${COLLECT_DIR:-$HOME/workspace/physvla_sim/collect}"
TASK_ID="${TASK_ID:-adjust_the_monitor}"
WORKER_ID="${WORKER_ID:-main}"
DATASET_ROOT="${DATASET_ROOT:-$COLLECT_DIR/datasets}"
DATASET_DIR="${DATASET_DIR:-$DATASET_ROOT/$TASK_ID}"
DATASET_STEM="${DATASET_STEM:-$TASK_ID}"
DATASET_FILE="${DATASET_FILE:-$DATASET_DIR/${DATASET_STEM}.hdf5}"
# This is the remaining target for this run/batch, not the total historical dataset size.
TARGET_DEMOS="${TARGET_DEMOS:-300}"
BATCH_SIZE="${BATCH_SIZE:-300}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"
EPISODE_STEP_LIMIT="${EPISODE_STEP_LIMIT:-800}"
SEED="${SEED:-42}"
RESTART_SLEEP_SEC="${RESTART_SLEEP_SEC:-10}"
POST_RESET_WARMUP_SEC="${POST_RESET_WARMUP_SEC:-1.0}"
COLLECTOR_TIMEOUT_SEC="${COLLECTOR_TIMEOUT_SEC:-21600}"
STOP_AFTER_RUN_TARGET="${STOP_AFTER_RUN_TARGET:-1}"
STOP_AFTER_TARGET_GRACE_SEC="${STOP_AFTER_TARGET_GRACE_SEC:-20}"
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-5}"
LOG_ROOT="${LOG_ROOT:-$COLLECT_DIR/logs/watchdog}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$WORKER_ID}"
BASELINE_FILE="$LOG_DIR/baseline_files.txt"
BATCH_FILE="$LOG_DIR/current_batch_files.txt"
PID_FILE="$LOG_DIR/current_collector.pid"
mkdir -p "$DATASET_DIR" "$LOG_DIR"

snapshot_files() {
  find "$DATASET_DIR" -maxdepth 2 -name '*.hdf5' -printf '%p\n' | sort
}

count_demos_in_files() {
  "$HOME/isaacsim/python.sh" - "$BATCH_FILE" <<'PY'
import h5py
import sys
from pathlib import Path
state_file = Path(sys.argv[1])
total = 0
if state_file.exists():
    for line in state_file.read_text().splitlines():
        path = Path(line.strip())
        if not path.exists():
            continue
        try:
            with h5py.File(path, "r", locking=False) as f:
                total += len([k for k in f.get("data", {}).keys() if k.startswith("demo_")])
        except Exception:
            pass
print(total)
PY
}

record_new_files() {
  comm -13 "$BASELINE_FILE" <(snapshot_files) > "$BATCH_FILE" || true
}

kill_collect_tree() {
  if [[ -s "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
      sleep 2
      pkill -KILL -P "$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  pkill -9 -f "auto_trajectory_collection.py --task_id $TASK_ID .*--dataset_file $DATASET_FILE" 2>/dev/null || true
}

cd "$COLLECT_DIR"
snapshot_files > "$BASELINE_FILE"
: > "$BATCH_FILE"

echo "[WATCHDOG][$WORKER_ID] task=$TASK_ID target=$TARGET_DEMOS batch_size=$BATCH_SIZE seed=$SEED"
echo "[WATCHDOG][$WORKER_ID] dataset_file=$DATASET_FILE"
echo "[WATCHDOG][$WORKER_ID] log_dir=$LOG_DIR"

while true; do
  record_new_files
  current_batch="$(count_demos_in_files)"
  remaining=$((TARGET_DEMOS - current_batch))
  if (( remaining <= 0 )); then
    echo "[WATCHDOG][$WORKER_ID] Batch target reached: ${current_batch}/${TARGET_DEMOS}."
    exit 0
  fi

  run_demos=$remaining
  if (( run_demos > BATCH_SIZE )); then
    run_demos=$BATCH_SIZE
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/auto_collect_${WORKER_ID}_${ts}_remaining_${remaining}_run_${run_demos}.log"
  echo "[WATCHDOG][$WORKER_ID] Batch current=${current_batch}/${TARGET_DEMOS}, remaining=${remaining}, this_run=${run_demos}. Log: $log_file"

  flat_dataset_args=()
  if [[ "${FLAT_DATASET_SESSION_DIR:-0}" == "1" ]]; then
    flat_dataset_args=(--flat_dataset_session_dir)
  fi

  set +e
  "$HOME/isaacsim/python.sh" auto_trajectory_collection.py \
    --task_id "$TASK_ID" \
    --headless \
    --num_demos "$run_demos" \
    --max_attempts "$MAX_ATTEMPTS" \
    --episode_step_limit "$EPISODE_STEP_LIMIT" \
    --seed "$SEED" \
    --post_reset_warmup_sec "$POST_RESET_WARMUP_SEC" \
    --dataset_file "$DATASET_FILE" \
    "${flat_dataset_args[@]}" \
    > >(tee "$log_file") 2>&1 &
  collector_pid=$!
  echo "$collector_pid" > "$PID_FILE"
  exit_code=0
  timed_out=0
  target_reached_stop=0
  run_target=$((current_batch + run_demos))
  deadline=0
  if (( COLLECTOR_TIMEOUT_SEC > 0 )); then
    deadline=$((SECONDS + COLLECTOR_TIMEOUT_SEC))
  fi
  while kill -0 "$collector_pid" 2>/dev/null; do
    record_new_files
    observed_batch="$(count_demos_in_files)"
    if (( STOP_AFTER_RUN_TARGET != 0 && observed_batch >= run_target )); then
      echo "[WATCHDOG][$WORKER_ID] Run target reached (${observed_batch}/${run_target}); waiting ${STOP_AFTER_TARGET_GRACE_SEC}s before stopping this collector."
      sleep "$STOP_AFTER_TARGET_GRACE_SEC"
      target_reached_stop=1
      kill_collect_tree
      break
    fi
    if (( deadline > 0 && SECONDS >= deadline )); then
      echo "[WATCHDOG][$WORKER_ID] Collector timeout after ${COLLECTOR_TIMEOUT_SEC}s; killing this worker collector."
      timed_out=1
      kill_collect_tree
      break
    fi
    sleep "$WATCHDOG_POLL_SEC"
  done
  if (( timed_out == 0 && target_reached_stop == 0 )); then
    wait "$collector_pid"
    exit_code=$?
  else
    wait "$collector_pid" 2>/dev/null || true
    if (( timed_out != 0 )); then
      exit_code=124
    else
      exit_code=0
    fi
  fi
  rm -f "$PID_FILE"
  set -e

  record_new_files
  current_batch="$(count_demos_in_files)"
  remaining=$((TARGET_DEMOS - current_batch))
  if (( remaining <= 0 )); then
    echo "[WATCHDOG][$WORKER_ID] Batch target reached after run: ${current_batch}/${TARGET_DEMOS}."
    exit 0
  fi

  if rg -q "FATAL_WRIST_CAMERA_MOUNT|cudaErrorIllegalAddress|cudaErrorMisalignedAddress|illegal memory access|misaligned address|Failed to wait on external semaphore|Wait for external semaphore failed|out of memory|CUDA out of memory" "$log_file"; then
    echo "[WATCHDOG][$WORKER_ID] Fatal Isaac/CUDA/camera condition detected. Killing this worker collector and restarting after ${RESTART_SLEEP_SEC}s."
    kill_collect_tree
    sleep "$RESTART_SLEEP_SEC"
    continue
  fi

  if [[ "$exit_code" == "75" ]]; then
    echo "[WATCHDOG][$WORKER_ID] Collector requested restart (exit 75). Sleeping ${RESTART_SLEEP_SEC}s."
    kill_collect_tree
    sleep "$RESTART_SLEEP_SEC"
    continue
  fi

  if [[ "$exit_code" != "0" ]]; then
    echo "[WATCHDOG][$WORKER_ID] Collector exited with code $exit_code. Killing leftovers and restarting after ${RESTART_SLEEP_SEC}s."
    kill_collect_tree
    sleep "$RESTART_SLEEP_SEC"
    continue
  fi

  echo "[WATCHDOG][$WORKER_ID] Collector exited cleanly but batch target not reached (${current_batch}/${TARGET_DEMOS}); restarting after ${RESTART_SLEEP_SEC}s."
  sleep "$RESTART_SLEEP_SEC"
done
