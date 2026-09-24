#!/usr/bin/env python3
"""Acceptance checks 1-3 for the remote PRTS policy server (no Isaac Sim).

  1. connect + metadata matches the left6 contract
  2. one fake observation returns (50, 7) float32 with no NaN
  3. same obs + same seed is bit-identical; without seed it is not

Run with the Isaac python so the deps match the rollout process:
  cd /home/ubuntu/isaacsim && source setup_python_env.sh
  PYTHONPATH="/home/ubuntu/workspace/PRTS/third_party/openpi-client/src:$PYTHONPATH" \
    ./kit/python/bin/python3 <this file> --host 127.0.0.1 --port 10093
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

EXPECTED_METADATA = {
    "action_dim": 7,
    "action_chunk_size": 50,
    "embodiment_tag": "piper_left6",
    "state_relative_action": True,
    "state_mode": "QUANTILE",
}
CAMERA_ORDER = ("head", "left_wrist")


def build_fake_obs(prompt: str, seed: int | None = None) -> dict:
    """Deterministic fake observation in the server's wire format."""

    rng = np.random.default_rng(0)
    obs: dict = {}
    for name in CAMERA_ORDER:
        obs[f"observation/{name}"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    obs["observation/state"] = np.zeros(7, dtype=np.float32)
    obs["prompt"] = prompt
    if seed is not None:
        obs["seed"] = int(seed)
    return obs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--prompt", default="close the laptop lid.")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    from openpi_client import websocket_client_policy

    failures: list[str] = []

    # --- 1. connect + metadata -------------------------------------------
    print(f"[1] connecting to ws://{args.host}:{args.port} ...", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    print(f"[1] metadata: {json.dumps(meta, ensure_ascii=False, default=str)}")
    for key, want in EXPECTED_METADATA.items():
        got = meta.get(key)
        if got != want:
            failures.append(f"metadata.{key}: expected {want!r}, got {got!r}")
    print(f"[1] {'PASS' if not failures else 'FAIL'}")

    # --- 2. one fake observation -----------------------------------------
    print("[2] sending one fake observation ...", flush=True)
    t0 = time.perf_counter()
    out = client.infer(build_fake_obs(args.prompt))
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    actions = np.asarray(out["actions"])
    print(
        f"[2] actions shape={actions.shape} dtype={actions.dtype} "
        f"rtt={rtt_ms:.0f}ms server={out.get('server_timing')}"
    )
    check2 = []
    if actions.shape != (50, 7):
        check2.append(f"shape: expected (50, 7), got {actions.shape}")
    if actions.dtype != np.float32:
        check2.append(f"dtype: expected float32, got {actions.dtype}")
    if not np.all(np.isfinite(actions)):
        check2.append("actions contain NaN/Inf")
    per_dim_std = actions.std(axis=0)
    print(f"[2] per-dim std: {np.array2string(per_dim_std, precision=3)}")
    if abs(float(per_dim_std[6])) > 1e-6:
        check2.append(f"gripper dim std should be ~0, got {per_dim_std[6]:.6f}")
    failures.extend(check2)
    print(f"[2] {'PASS' if not check2 else 'FAIL'}")

    # --- 3. seed reproducibility -----------------------------------------
    print(f"[3] same obs twice with seed={args.seed} ...", flush=True)
    a1 = np.asarray(client.infer(build_fake_obs(args.prompt, args.seed))["actions"])
    a2 = np.asarray(client.infer(build_fake_obs(args.prompt, args.seed))["actions"])
    seeded_maxdiff = float(np.abs(a1 - a2).max())
    b1 = np.asarray(client.infer(build_fake_obs(args.prompt))["actions"])
    b2 = np.asarray(client.infer(build_fake_obs(args.prompt))["actions"])
    unseeded_maxdiff = float(np.abs(b1 - b2).max())
    print(f"[3] seeded maxdiff   = {seeded_maxdiff:.9f} (expect 0.0)")
    print(f"[3] unseeded maxdiff = {unseeded_maxdiff:.9f} (expect > 0, flow-matching noise)")
    check3 = []
    if seeded_maxdiff != 0.0:
        check3.append(f"seeded inference not bit-identical: maxdiff={seeded_maxdiff:.9f}")
    if unseeded_maxdiff == 0.0:
        check3.append("unseeded inference is identical; server may ignore the noise seed entirely")
    failures.extend(check3)
    print(f"[3] {'PASS' if not check3 else 'FAIL'}")

    # The sync websocket keeps a non-daemon reader alive; close it or we hang on exit.
    try:
        client._ws.close()
    except Exception:
        pass

    print("")
    if failures:
        print("SELFCHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELFCHECK PASSED (checks 1-3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
