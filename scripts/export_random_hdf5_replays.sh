#!/bin/bash
# Randomly sample N demos from each HDF5 and export side-by-side camera MP4s.
set -euo pipefail

HDF5_DIR="${1:-$HOME/workspace/physvla_sim/collect/datasets/close_laptop_lid}"
OUT_ROOT="${2:-$HOME/workspace/physvla_sim/hdf5_replays/224*224_laptop_random_replay}"
SAMPLES_PER_FILE="${3:-5}"
SEED="${4:-42}"

PYTHON="$HOME/isaacsim/python.sh"
VIEWER="$HOME/workspace/physvla_sim/hdf5_replays/view_hdf5_cameras.py"
FFMPEG="/usr/bin/ffmpeg"

mkdir -p "$OUT_ROOT"

mapfile -t HDF5_FILES < <(find "$HDF5_DIR" -maxdepth 1 -name '*.hdf5' | sort)
if ((${#HDF5_FILES[@]} == 0)); then
  echo "[ERROR] No HDF5 files under: $HDF5_DIR" >&2
  exit 1
fi

echo "[INFO] HDF5 dir: $HDF5_DIR"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Samples per file: $SAMPLES_PER_FILE (seed=$SEED)"
echo "[INFO] Files: ${#HDF5_FILES[@]}"

for hdf5 in "${HDF5_FILES[@]}"; do
  stem="$(basename "${hdf5%.hdf5}")"
  out_dir="$OUT_ROOT/$stem"
  mkdir -p "$out_dir"

  mapfile -t PICKED < <(
    "$PYTHON" - <<PY
import random
import h5py
from pathlib import Path

path = Path("$hdf5")
with h5py.File(path, "r", locking=False) as f:
    demos = sorted(
        [k for k in f["data"].keys() if k.startswith("demo_")],
        key=lambda x: int(x.split("_")[1]),
    )
if not demos:
    raise SystemExit(0)
random.seed(int("$SEED") + hash("$stem") % 100000)
k = min(int("$SAMPLES_PER_FILE"), len(demos))
for name in sorted(random.sample(demos, k), key=lambda x: int(x.split("_")[1])):
    print(name)
PY
  )

  if ((${#PICKED[@]} == 0)); then
    echo "[WARN] Skip empty: $hdf5"
    continue
  fi

  echo "[INFO] $stem -> ${#PICKED[@]} demos: ${PICKED[*]}"
  for demo in "${PICKED[@]}"; do
    success_tag="unk"
    if tag=$("$PYTHON" - <<PY
import h5py
from pathlib import Path
with h5py.File(Path("$hdf5"), "r", locking=False) as f:
    ep = f["data"]["$demo"]
    success = ep.attrs.get("success", None)
if success in (True, 1, "True", "true"):
    print("ok")
elif success in (False, 0, "False", "false"):
    print("fail")
else:
    print("unk")
PY
    ); then
      success_tag="$tag"
    fi
    out_mp4="$out_dir/${demo}_${success_tag}.mp4"
    if [[ -f "$out_mp4" ]]; then
      echo "  [SKIP] exists: $out_mp4"
      continue
    fi
    echo "  [EXPORT] $demo -> $out_mp4"
    "$PYTHON" "$VIEWER" \
      --file "$hdf5" \
      --demo "$demo" \
      --mode h264 \
      --fps 30 \
      --stride 1 \
      --encoder nvenc \
      --ffmpeg "$FFMPEG" \
      --output-h264 "$out_mp4"
  done
done

echo "[INFO] Done. Output: $OUT_ROOT"
