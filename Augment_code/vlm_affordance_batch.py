#!/usr/bin/env python3
"""Batch VLM inference for affordance position_xyz on FITR-Bench.

Reproduce (cd physvla_sim/Augment_code; export DASHSCOPE_API_KEY='sk-...'):

  # Preview prompt for one asset
  .venv/bin/python -m affordance.vlm_affordance_prompt --category Door --asset-id 9280

  # Bench subset (dry-run)
  .venv/bin/python vlm_affordance_batch.py --category Door --bench --dry-run

  # Run VLM affordance baseline on full bench
  .venv/bin/python vlm_affordance_batch.py --bench --sleep 1.0

  # Evaluate position_xyz error (cm) vs bench GT
  .venv/bin/python eval_affordance_vlm_bench.py --report affordance_vlm_bench
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUGMENT_ROOT = Path(__file__).resolve().parent
if str(AUGMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUGMENT_ROOT))

from affordance.vlm_affordance_prompt import (
    build_prompt_for_bench_asset,
    load_bench_assets,
)
from experiment_paths import DEFAULT_VLM_AFFORDANCE_RESULTS, ensure_experiment_dirs
from prompt.load_prompt import DEFAULT_BENCH_PATH
from vlm_batch import append_jsonl, call_vlm, parse_vlm_json

DEFAULT_MODEL = "qwen-vl-plus"


def parse_affordance_vlm_json(text: str) -> dict[str, Any]:
    """Parse VLM JSON; repair common invalid keys like {joint_0: ...}."""
    import re

    try:
        return parse_vlm_json(text)
    except json.JSONDecodeError:
        repaired = re.sub(
            r'([{\[,]\s*)(joint[\w]*)\s*:',
            r'\1"\2":',
            text,
        )
        return parse_vlm_json(repaired)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_affordance_result(
    parsed: dict[str, Any],
    expected_joints: list[str],
) -> dict[str, Any]:
    raw = parsed.get("affordances") or {}
    if not isinstance(raw, dict):
        raise ValueError("affordances must be an object keyed by joint_name")
    out: dict[str, dict[str, Any]] = {}
    for joint_name in expected_joints:
        entry = raw.get(joint_name)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing affordances[{joint_name!r}]")
        pos = entry.get("position_xyz")
        if not isinstance(pos, list) or len(pos) != 3:
            raise ValueError(f"affordances[{joint_name}].position_xyz must be [x,y,z]")
        out[joint_name] = {
            "position_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
            "confidence": float(entry["confidence"]) if entry.get("confidence") is not None else None,
        }
    confidence = parsed.get("confidence")
    return {
        "affordances": out,
        "confidence": float(confidence) if confidence is not None else None,
    }


def load_done_keys(path: Path, *, model: str | None = None) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                if model is not None and rec.get("model") != model:
                    continue
                done.add((rec["category"], str(rec["asset_id"])))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM batch for affordance position_xyz.")
    parser.add_argument("--output", type=Path, default=DEFAULT_VLM_AFFORDANCE_RESULTS)
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH_PATH)
    parser.add_argument("--bench", action="store_true", help="Only FITR-Bench assets")
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--asset-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.asset_id and (not args.category or len(args.category) != 1):
        parser.error("--asset-id requires exactly one --category")

    ensure_experiment_dirs()
    work = load_bench_assets(args.bench_path, categories=args.category)
    if args.asset_id:
        asset_ids = {str(x) for x in args.asset_id}
        work = [(c, a) for c, a in work if str(a["asset_id"]) in asset_ids]
    elif not args.bench:
        parser.error("Use --bench (FITR-Bench only) or pass --category + --asset-id")

    done = set() if args.force else load_done_keys(args.output, model=args.model)
    pending = [(c, a) for c, a in work if (c, str(a["asset_id"])) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"planned={len(work)} pending={len(pending)} output={args.output}")
    if not pending:
        return 0

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not args.dry_run and not api_key:
        print("DASHSCOPE_API_KEY is not set.", file=sys.stderr)
        return 1

    ok = err = 0
    for category, asset in pending:
        asset_id = str(asset["asset_id"])
        try:
            bundle = build_prompt_for_bench_asset(
                category,
                asset,
            )
        except Exception as exc:
            print(f"[error] {category}/{asset_id} prompt: {exc}", file=sys.stderr)
            err += 1
            continue

        joint_names = [a.joint_name for a in bundle.affordances]
        print(
            f"[{'dry' if args.dry_run else 'run'}] {category}/{asset_id} "
            f"joints={joint_names} image={bundle.image_path.name}"
        )
        if args.dry_run:
            ok += 1
            continue

        record: dict[str, Any] = {
            "category": category,
            "asset_id": asset_id,
            "model": args.model,
            "task": "affordance_position_xyz",
            "image": str(bundle.image_path.relative_to(bundle.asset_dir)),
            "joint_names": joint_names,
            "timestamp": utc_now(),
        }
        raw: str | None = None
        try:
            raw = call_vlm(
                system=bundle.system,
                user=bundle.user,
                image_paths=[bundle.image_path],
                model=args.model,
                api_key=api_key,
            )
            parsed = normalize_affordance_result(
                parse_affordance_vlm_json(raw), joint_names
            )
            record.update(
                status="ok",
                raw_response=raw,
                result=parsed,
                error=None,
            )
            ok += 1
            print(f"[ok] {category}/{asset_id} {parsed['affordances']}")
        except Exception as exc:
            record.update(
                status="error",
                raw_response=raw,
                result=None,
                error=str(exc),
            )
            err += 1
            print(f"[error] {category}/{asset_id}: {exc}", file=sys.stderr)

        append_jsonl(args.output, record)
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"finished ok={ok} errors={err}")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
