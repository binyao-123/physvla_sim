#!/usr/bin/env bash
set -euo pipefail

USD_LIBS="$(ls -d "$HOME"/isaacsim/extscache/omni.usd.libs-* | head -n 1)"
cd "$(dirname "$0")/.."

PYTHONPATH="$USD_LIBS${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH="$USD_LIBS/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$HOME/isaacsim/kit/python/bin/python3" scripts/download_123_assets.py "$@"
