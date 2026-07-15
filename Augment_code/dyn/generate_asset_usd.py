#!/usr/bin/env python3
"""Convert an asset URDF, then persist info1.json into the layered USD files.

The conversion and authoring steps intentionally run in separate Isaac Sim
processes. The URDF converter finalizes its layers during shutdown, so editing
those layers in the conversion process would be silently overwritten.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def find_urdf(asset_dir: Path, explicit_name: str | None) -> Path:
    if explicit_name:
        path = asset_dir / explicit_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    preferred = [asset_dir / "mobility.urdf", asset_dir / "test.urdf"]
    matches = [path for path in preferred if path.is_file()]
    if not matches:
        matches = sorted(asset_dir.glob("*.urdf"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one auto-detectable URDF in {asset_dir}, "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def validate_urdf(urdf_path: Path) -> None:
    robot = ET.parse(urdf_path).getroot()
    links = [link.get("name") for link in robot.findall("link")]
    duplicate_links = sorted({name for name in links if links.count(name) > 1})
    self_joints = [
        joint.get("name")
        for joint in robot.findall("joint")
        if joint.find("parent").get("link") == joint.find("child").get("link")
    ]
    errors = []
    if duplicate_links:
        errors.append(f"duplicate links: {', '.join(duplicate_links)}")
    if self_joints:
        errors.append(f"self-referencing joints: {', '.join(self_joints)}")
    if errors:
        raise ValueError(f"Invalid URDF {urdf_path.name}: {'; '.join(errors)}")


def sanitize_urdf_structure(urdf_path: Path) -> Path:
    """Create a temporary URDF with duplicate links renamed for converter safety."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    asset_dir = urdf_path.parent
    duplicate_links: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    changed = False

    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count == 0:
            continue
        safe_name = f"{name}__dup{count}"
        link.set("name", safe_name)
        duplicate_links.setdefault(name, []).append(safe_name)
        changed = True

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if parent_name == child_name and child_name in duplicate_links and duplicate_links[child_name]:
            child.set("link", duplicate_links[child_name].pop(0))
            changed = True

    if not changed:
        return urdf_path
    safe_path = asset_dir / f".{urdf_path.stem}_usd_safe.urdf"
    tree.write(safe_path, encoding="unicode", xml_declaration=True)
    return safe_path


def sanitize_urdf_mesh_paths(urdf_path: Path) -> Path:
    """Rewrite mesh refs whose basename contains '-' (invalid USD prim paths)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    asset_dir = urdf_path.parent
    changed = False
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        rel = Path(filename)
        if "-" not in rel.stem:
            continue
        safe_name = rel.name.replace("-", "_")
        safe_rel = str(rel.with_name(safe_name)).replace("\\", "/")
        source = asset_dir / filename
        target = asset_dir / safe_rel
        if source.is_file() and not target.exists():
            target.symlink_to(source.resolve())
        mesh.set("filename", safe_rel)
        changed = True
    if not changed:
        return urdf_path
    safe_path = asset_dir / f".{urdf_path.stem}_usd_safe.urdf"
    tree.write(safe_path, encoding="unicode", xml_declaration=True)
    return safe_path


def validate_convert_output(asset_dir: Path, output_name: str) -> None:
    stem = Path(output_name).stem
    base_path = asset_dir / "configuration" / f"{stem}_base.usd"
    if not base_path.is_file() or base_path.stat().st_size < 1000:
        raise RuntimeError(f"Convert produced incomplete base layer: {base_path}")


def validate_author_output(asset_dir: Path, output_name: str, stdout: str) -> None:
    if '"joints"' not in stdout:
        raise RuntimeError("Author did not emit verified joint output")
    stem = Path(output_name).stem
    physics_path = asset_dir / "configuration" / f"{stem}_physics.usd"
    if not physics_path.is_file() or physics_path.stat().st_size < 500:
        raise RuntimeError(f"Author produced incomplete physics layer: {physics_path}")


def run_isaac_subprocess(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def remove_outputs(asset_dir: Path, output_name: str) -> None:
    output_path = asset_dir / output_name
    stem = output_path.stem
    generated = [output_path, *(asset_dir / "configuration" / f"{stem}_{kind}.usd"
                                for kind in ("base", "physics", "robot", "sensor"))]
    for path in generated:
        path.unlink(missing_ok=True)
    configuration_dir = asset_dir / "configuration"
    if configuration_dir.exists() and not any(configuration_dir.iterdir()):
        configuration_dir.rmdir()
    (asset_dir / ".asset_hash").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_dir", type=Path)
    parser.add_argument("--urdf-name")
    parser.add_argument("--output-name", default="fitr_info1.usd")
    parser.add_argument("--no-fix-base", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=int, default=120, help="Per-process timeout in seconds.")
    args = parser.parse_args()

    asset_dir = args.asset_dir.expanduser().resolve()
    if not asset_dir.is_dir():
        raise NotADirectoryError(asset_dir)
    if not (asset_dir / "info1.json").is_file():
        raise FileNotFoundError(asset_dir / "info1.json")

    urdf_path = find_urdf(asset_dir, args.urdf_name)
    output_path = asset_dir / args.output_name
    urdf_path = sanitize_urdf_structure(urdf_path)
    validate_urdf(urdf_path)
    urdf_path = sanitize_urdf_mesh_paths(urdf_path)
    workspace_root = Path(__file__).resolve().parents[3]
    converter = workspace_root / "IsaacLab" / "scripts" / "tools" / "convert_urdf.py"
    author = Path(__file__).with_name("author_info1_usd.py")
    isaac_root = Path(os.environ.get("ISAAC_PATH", Path.home() / "isaacsim"))
    isaac_python = isaac_root / "python.sh"

    convert_command = [
        str(isaac_python),
        str(converter),
        str(urdf_path),
        str(output_path),
        "--joint-stiffness",
        "0",
        "--joint-damping",
        "0",
        "--joint-target-type",
        "position",
    ]
    if not args.no_fix_base:
        convert_command.append("--fix-base")
    if args.headless:
        convert_command.append("--headless")
    try:
        run_isaac_subprocess(convert_command, args.timeout)
        validate_convert_output(asset_dir, args.output_name)
        author_command = [
            str(isaac_python),
            str(author),
            str(asset_dir),
            "--output-name",
            args.output_name,
        ]
        if args.headless:
            author_command.append("--headless")
        author_result = run_isaac_subprocess(author_command, args.timeout)
        validate_author_output(asset_dir, args.output_name, author_result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError, ValueError):
        remove_outputs(asset_dir, args.output_name)
        raise


if __name__ == "__main__":
    main()
