#!/usr/bin/env python3
"""Normalized Joint Completion (NJC) for a rollout output directory.

FITR 5.1:  C_r = clip(1 - |q_r^T - q_r^g| / |q_r^0 - q_r^g|, 0, 1)

q^T is the *terminal* joint position. The terminal value is the metric; the peak
is kept only as a diagnostic. The two differ exactly when the arm drives the joint
part way and then loses it -- FITR 5.5 lists "premature termination at intermediate
joint states" as a main failure mode, and scoring on the peak would hide it.

Overshoot past the target still scores 1.0: an episode terminates as soon as the
joint enters the success threshold, so a clean success lands slightly beyond it,
and FITR F.4 defines C_r = 1 as "the target joint reaches the task target".

  python3 rollout_sim/eval_njc.py <rollout_output_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Report an episode as slip-back when peak beat terminal by more than this (percentage points).
SLIP_BACK_REPORT_PCT = 1.0


def _initial_deg(episode: dict, joint_prim: str) -> float | None:
    for spec in episode.get("joint_initial_specs") or []:
        if spec.get("prim_path") == joint_prim:
            return float(spec["position"])
    return None


def _completion_pct(init: float, value: float, target: float) -> float:
    """Signed progress from init toward target, clipped to [0, 100]."""

    denom = target - init
    if abs(denom) < 1e-6:
        return 100.0 if abs(value - target) < 1e-3 else 0.0
    return max(0.0, min(100.0, (value - init) / denom * 100.0))


def score_episode(episode: dict, joint_prim: str, target_deg: float) -> dict | None:
    init = _initial_deg(episode, joint_prim)
    final = (episode.get("final_joint_degs") or {}).get(joint_prim)
    if init is None or final is None:
        return None
    peak = (episode.get("peak_joint_degs") or {}).get(joint_prim)
    njc = _completion_pct(init, float(final), target_deg)
    njc_peak = _completion_pct(init, float(peak), target_deg) if peak is not None else None
    return {
        "episode_index": episode.get("episode_index"),
        "success": bool(episode.get("success")),
        "njc_pct": njc,
        "njc_peak_pct": njc_peak,
        "slip_back_pct": None if njc_peak is None else max(0.0, njc_peak - njc),
        "initial_deg": init,
        "final_deg": float(final),
        "peak_deg": None if peak is None else float(peak),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: eval_njc.py <rollout_output_dir>")
        return 2
    output_dir = Path(sys.argv[1])
    summary_path = output_dir / "summary.json"
    meta_path = output_dir / "run_meta.json"
    if not summary_path.is_file() or not meta_path.is_file():
        print(f"[EVAL] missing summary/meta under {output_dir}")
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    specs = meta.get("rollout_success_specs") or []
    if not specs:
        print("[EVAL] rollout_success_specs empty; cannot score NJC")
        return 2

    joint_prim = specs[0]["joint_prim"]
    if specs[0].get("angle_gt_deg") is not None:
        target = float(specs[0]["angle_gt_deg"])
    elif specs[0].get("angle_lt_deg") is not None:
        target = float(specs[0]["angle_lt_deg"])
    else:
        print("[EVAL] no angle_gt_deg/angle_lt_deg in success spec")
        return 2

    rows = []
    for episode in summary.get("episode_results") or []:
        row = score_episode(episode, joint_prim, target)
        if row is None:
            print(f"[EVAL] episode_{episode.get('episode_index', -1):04d}: no joint readout, skipped")
            continue
        rows.append(row)

    if not rows:
        print("[EVAL] No scorable episodes.")
        return 2

    print("")
    print("[EVAL] -------- per-episode NJC (terminal) --------")
    for row in rows:
        slip = row["slip_back_pct"]
        note = ""
        if slip is not None and slip > SLIP_BACK_REPORT_PCT:
            note = f"   <- slipped back {slip:.1f}pp from peak {row['njc_peak_pct']:.1f}%"
        print(f"[EVAL] episode_{row['episode_index']:04d}: {row['njc_pct']:.1f}%{note}")

    njc_values = [r["njc_pct"] for r in rows]
    peak_values = [r["njc_peak_pct"] for r in rows if r["njc_peak_pct"] is not None]
    mean_njc = sum(njc_values) / len(njc_values)
    mean_peak = sum(peak_values) / len(peak_values) if peak_values else None
    slipped = [r for r in rows if (r["slip_back_pct"] or 0.0) > SLIP_BACK_REPORT_PCT]

    result = {
        "metric": "NJC (FITR 5.1), terminal joint position",
        "num_episodes": len(rows),
        "mean_njc_pct": mean_njc,
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "mean_njc_peak_pct_diagnostic": mean_peak,
        "num_slip_back_episodes": len(slipped),
        "episodes": rows,
        "output_dir": str(output_dir),
    }
    out_json = output_dir / "eval_progress_summary.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("")
    print(f"[EVAL] mean NJC over {len(rows)} episodes: {mean_njc:.1f}%   (success rate {result['success_rate']*100:.1f}%)")
    if mean_peak is not None:
        print(f"[EVAL] diagnostic: mean peak-based score {mean_peak:.1f}%, {len(slipped)} episode(s) slipped back")
    print(f"[EVAL] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
