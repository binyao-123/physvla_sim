#!/bin/bash
# One-time: Pi05 rollout deps inside Isaac Sim Python only (does NOT touch lerobot uv/.venv).
set -euo pipefail

PY=~/isaacsim/python.sh

echo "[INFO] Isaac Python: $($PY -V)"
echo "[INFO] Installing draccus + transformers (Pi05 inference)..."

$PY -m pip install 'draccus==0.10.0' 'transformers>=5.4.0,<5.6.0'

echo "[INFO] Verify imports..."
$PY -c "
import sys
sys.path.insert(0, '/home/ubuntu/workspace/lerobot/src')
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor.pipeline import PolicyProcessorPipeline
print('lerobot PI05 import OK')
"

echo "[INFO] Done. Note: isaaclab may pin transformers==4.57.6; rollout needs 5.x for Pi05."
echo "[INFO] Training still uses: cd ~/workspace/lerobot && uv run ..."
