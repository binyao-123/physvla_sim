#!/usr/bin/env python3
"""Align PI05 checkpoint metadata with the local Isaac rollout runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import fields
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = WORKSPACE_ROOT / "lerobot" / "src"
if not LEROBOT_SRC.is_dir():
    raise FileNotFoundError(f"LeRobot source directory not found: {LEROBOT_SRC}")
sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.policies.pi05.configuration_pi05 import PI05Config


parser = argparse.ArgumentParser(
    description="Align PI05 checkpoint config/preprocessor files with the local runtime."
)
parser.add_argument(
    "--policy-path",
    required=True,
    type=Path,
    help="Checkpoint pretrained_model directory containing config.json.",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Report incompatible fields without changing files.",
)
parser.add_argument(
    "--tokenizer-name",
    default=None,
    help=(
        "Tokenizer path/id to write into policy_preprocessor.json. "
        "Default: local HF cache snapshot for google/paligemma-3b-pt-224 if present, "
        "else the Hub id."
    ),
)
args = parser.parse_args()

policy_path = args.policy_path.expanduser().resolve()

HUB_TOKENIZER_ID = "google/paligemma-3b-pt-224"


def resolve_local_tokenizer_dir(repo_id: str = HUB_TOKENIZER_ID) -> Path | None:
    """Return a local HF hub snapshot dir that contains tokenizer files, if cached."""

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for snap in sorted(snapshots.iterdir(), reverse=True):
        if not snap.is_dir():
            continue
        if (snap / "tokenizer.json").is_file() or (snap / "tokenizer_config.json").is_file():
            return snap
    return None


_local_tokenizer = resolve_local_tokenizer_dir()
if args.tokenizer_name:
    tokenizer_target = str(Path(args.tokenizer_name).expanduser())
else:
    tokenizer_target = str(_local_tokenizer) if _local_tokenizer is not None else HUB_TOKENIZER_ID
print(f"[INFO] Tokenizer target for align: {tokenizer_target}")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def write_json_atomic(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(data, temp_file, indent=4, ensure_ascii=False)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


changed_files: list[str] = []

config_path = policy_path / "config.json"
if not config_path.is_file():
    raise FileNotFoundError(f"PI05 config not found: {config_path}")

config = read_json(config_path)
if config.get("type") != "pi05":
    raise ValueError(f"Expected a PI05 config, got type={config.get('type')!r} in {config_path}")

supported_fields = {field.name for field in fields(PI05Config)}
supported_fields.add("type")
unsupported_fields = sorted(set(config) - supported_fields)

if unsupported_fields:
    print(f"[INFO] Unsupported PI05 config fields: {', '.join(unsupported_fields)}")
    config = {key: value for key, value in config.items() if key in supported_fields}
    changed_files.append(str(config_path))
else:
    print(f"[INFO] PI05 config already compatible: {config_path}")

if str(config_path) in changed_files and not args.dry_run:
    write_json_atomic(config_path, config)

preprocessor_path = policy_path / "policy_preprocessor.json"
if preprocessor_path.is_file():
    preprocessor = read_json(preprocessor_path)
    tokenizer_updates = 0
    for step in preprocessor.get("steps", []):
        if not isinstance(step, dict) or step.get("registry_name") != "tokenizer_processor":
            continue
        step_config = step.get("config")
        if not isinstance(step_config, dict):
            continue
        tokenizer_name = str(step_config.get("tokenizer_name", ""))
        # Always pin to a resolvable local/Hub target so Isaac rollout does not
        # hit the network during AutoTokenizer init (SSL/proxy flakes).
        if tokenizer_name != tokenizer_target:
            print(
                "[INFO] Rewriting tokenizer_name: "
                f"{tokenizer_name} -> {tokenizer_target}"
            )
            step_config["tokenizer_name"] = tokenizer_target
            tokenizer_updates += 1
    if tokenizer_updates:
        if not args.dry_run:
            write_json_atomic(preprocessor_path, preprocessor)
            changed_files.append(str(preprocessor_path))
    else:
        print(f"[INFO] PI05 preprocessor already compatible: {preprocessor_path}")
else:
    print(f"[WARN] PI05 preprocessor not found: {preprocessor_path}")

postprocessor_path = policy_path / "policy_postprocessor.json"
if postprocessor_path.is_file():
    print(f"[INFO] PI05 postprocessor left unchanged: {postprocessor_path}")
else:
    print(f"[WARN] PI05 postprocessor not found: {postprocessor_path}")

if args.dry_run:
    raise SystemExit(0)
if changed_files:
    print("[INFO] Aligned PI05 metadata:")
    for file_path in changed_files:
        print(f"  - {file_path}")
else:
    print("[INFO] No PI05 metadata changes needed.")
