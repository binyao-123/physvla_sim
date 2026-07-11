"""Load and assemble vlm_prompt.txt for batch VLM inference."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scale.pixel_calibrate import find_scale_image, resolve_scale_method

PROMPT_DIR = Path(__file__).resolve().parent
AUGMENT_ROOT = PROMPT_DIR.parent
DEFAULT_PROMPT_PATH = PROMPT_DIR / "vlm_prompt.txt"
DEFAULT_VLM_BASE_PATH = AUGMENT_ROOT / "vlm_base_template.json"
DEFAULT_BENCH_PATH = AUGMENT_ROOT / "fitr_bench.json"
DEFAULT_DATASET_ROOT = AUGMENT_ROOT.parent / "datasets" / "data_normalized"
DEFAULT_CALIBRATION_CONFIG = AUGMENT_ROOT / "scale" / "calibration_config.json"

PARTNET_JOINT_MAP = {
    "hinge": "revolute",
    "slider": "prismatic",
    "rotation": "revolute",
    "translation": "prismatic",
}
SKIP_JOINTS = {"free", "heavy", "static", "fixed"}


@dataclass(frozen=True)
class JointSpec:
    joint_name: str
    link_name: str
    motion_type: str


@dataclass(frozen=True)
class ScaleReference:
    asset_id: str
    l_real_cm: float
    image_path: Path


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str
    category: str
    asset_id: str
    asset_dir: Path
    image_path: Path
    vlm_tasks: list[str]
    joints: tuple[JointSpec, ...]
    scale_refs: tuple[ScaleReference, ...] = ()
    vlm_scale_refs: tuple[ScaleReference, ...] = ()


_SECTION_RE = re.compile(r"^\[([A-Z0-9_]+)\]\s*$")


def parse_prompt_file(path: Path | None = None) -> dict[str, str]:
    path = Path(path or DEFAULT_PROMPT_PATH)
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        match = _SECTION_RE.match(line.strip())
        if match:
            current = match.group(1)
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def load_vlm_base(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or DEFAULT_VLM_BASE_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def _joints_from_info(info: dict[str, Any]) -> list[JointSpec]:
    joints: list[JointSpec] = []
    for link in info.get("links", []):
        if not link.get("motion_type"):
            continue
        joints.append(
            JointSpec(
                joint_name=f"joint_{len(joints)}",
                link_name=link["name"],
                motion_type=link["motion_type"],
            )
        )
    return joints


def _joints_from_mobility_v2(data: list[dict[str, Any]]) -> list[JointSpec]:
    joints: list[JointSpec] = []
    for item in data:
        jt = item.get("joint", "")
        if jt in SKIP_JOINTS:
            continue
        motion = PARTNET_JOINT_MAP.get(jt)
        if not motion:
            continue
        joints.append(
            JointSpec(
                joint_name=f"joint_{len(joints)}",
                link_name=item.get("name", f"link_{len(joints)}"),
                motion_type=motion,
            )
        )
    return joints


def _joints_from_urdf(asset_dir: Path) -> list[JointSpec]:
    for urdf_name in ("test.urdf", "mobility.urdf"):
        urdf_path = asset_dir / urdf_name
        if not urdf_path.is_file():
            continue
        tree = ET.parse(urdf_path)
        joints: list[JointSpec] = []
        for joint in tree.findall("joint"):
            jtype = joint.get("type")
            if jtype not in ("revolute", "prismatic", "continuous"):
                continue
            motion = "prismatic" if jtype == "prismatic" else "revolute"
            child = joint.find("child")
            if child is None:
                continue
            name = joint.get("name") or f"joint_{len(joints)}"
            joints.append(
                JointSpec(
                    joint_name=name,
                    link_name=child.get("link", f"link_{len(joints)}"),
                    motion_type=motion,
                )
            )
        if joints:
            return joints
    return []


def list_movable_joints(asset_dir: Path) -> list[JointSpec]:
    asset_dir = Path(asset_dir)
    info_path = asset_dir / "info.json"
    if info_path.is_file():
        joints = _joints_from_info(json.loads(info_path.read_text(encoding="utf-8")))
        if joints:
            return joints

    mob_path = asset_dir / "mobility_v2.json"
    if mob_path.is_file():
        joints = _joints_from_mobility_v2(json.loads(mob_path.read_text(encoding="utf-8")))
        if joints:
            return joints

    return _joints_from_urdf(asset_dir)


def load_calibration_config(path: Path | None = None) -> dict[str, Any]:
    path = Path(path or DEFAULT_CALIBRATION_CONFIG)
    if not path.is_file():
        return {"categories": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def find_reference_image(asset_dir: Path) -> Path:
    """Image for VLM affordance / door-type tasks."""
    asset_dir = Path(asset_dir)
    scale_image = find_scale_image(asset_dir)
    if scale_image is not None:
        return scale_image

    info_path = asset_dir / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        whole_image = info.get("whole_image")
        if whole_image:
            candidate = asset_dir / "images" / whole_image
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(f"No reference image under {asset_dir / 'images'}")


def find_category_scale_image(asset_dir: Path, category: str, calib: dict[str, Any] | None = None) -> Path:
    calib = calib or load_calibration_config()
    cfg = calib.get("categories", {}).get(category, {})
    preferred = cfg.get("image")
    asset_dir = Path(asset_dir)

    if preferred:
        for candidate in (asset_dir / preferred, asset_dir / "images" / preferred):
            if candidate.is_file():
                return candidate

    if category in {"Lamp", "Drawer"}:
        for fallback in (
            asset_dir / "parts_render" / "0.png",
            asset_dir / "parts_render_after_merging" / "0.png",
        ):
            if fallback.is_file():
                return fallback

    scale_image = find_scale_image(asset_dir)
    if scale_image is not None:
        return scale_image
    return find_reference_image(asset_dir)


def load_scale_references(
    category: str,
    category_dir: str,
    *,
    dataset_root: Path,
    bench_path: Path | None = None,
    calib: dict[str, Any] | None = None,
) -> tuple[ScaleReference, ...]:
    bench_path = Path(bench_path or DEFAULT_BENCH_PATH)
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    if category not in bench["categories"]:
        raise KeyError(f"Category {category} not found in {bench_path}")

    refs: list[ScaleReference] = []
    for asset in bench["categories"][category]["assets"]:
        asset_id = str(asset["asset_id"])
        l_real = asset["scale"]["L_real_cm_gt"]
        if l_real is None:
            raise ValueError(f"L_real_cm_gt missing for bench asset {category}/{asset_id}")
        asset_dir = dataset_root / category_dir / asset_id
        if not asset_dir.is_dir():
            raise FileNotFoundError(f"Bench scale reference asset not found: {asset_dir}")
        refs.append(
            ScaleReference(
                asset_id=asset_id,
                l_real_cm=float(l_real),
                image_path=find_category_scale_image(asset_dir, category, calib),
            )
        )
    if not refs:
        raise ValueError(f"No scale references in bench for category {category}")
    return tuple(refs)


def bench_scale_gt(asset_id: str, scale_refs: tuple[ScaleReference, ...]) -> float | None:
    for ref in scale_refs:
        if ref.asset_id == asset_id:
            return ref.l_real_cm
    return None


def vlm_scale_refs_for_target(
    scale_refs: tuple[ScaleReference, ...],
    target_asset_id: str,
    *,
    leave_one_out: bool = True,
) -> tuple[ScaleReference, ...]:
    """Refs sent to VLM: LOO when target is a bench ref; sorted smallest→largest."""
    refs = list(scale_refs)
    if leave_one_out:
        refs = [ref for ref in refs if ref.asset_id != target_asset_id]
    refs.sort(key=lambda ref: ref.l_real_cm)
    return tuple(refs)


def _render(template: str, **values: str) -> str:
    out = template
    for key, val in values.items():
        out = out.replace(f"{{{key}}}", val)
    return out


def _format_joint_lines(template: str, joints: list[JointSpec]) -> str:
    return "\n".join(
        _render(
            template,
            joint_name=j.joint_name,
            link_name=j.link_name,
            motion_type=j.motion_type,
        )
        for j in joints
    )


def _format_scale_ref_line(
    sections: dict[str, str],
    ref_index: int,
    ref: ScaleReference,
) -> str:
    return _render(
        sections["SCALE_REF_LINE"],
        ref_index=str(ref_index),
        ref_asset_id=ref.asset_id,
        ref_L_real_cm=str(ref.l_real_cm),
    )


def _build_scale_block(
    sections: dict[str, str],
    *,
    need_scale: bool,
    need_has_handle: bool,
    need_door_type: bool,
    vlm_scale_refs: tuple[ScaleReference, ...],
    asset_id: str,
) -> tuple[str, str]:
    if not need_scale:
        if need_has_handle and not need_door_type and sections.get("SCALE_BLOCK_HANDLE_ONLY", "").strip():
            return (
                _render(sections["SCALE_BLOCK_HANDLE_ONLY"], asset_id=asset_id),
                "1",
            )
        return (
            _render(
                sections["SCALE_BLOCK_NONE"],
                target_image_index="1",
                asset_id=asset_id,
            ),
            "1",
        )
    if not vlm_scale_refs:
        return sections["SCALE_BLOCK_NONE"], "1"

    ref_lines = "\n".join(
        _format_scale_ref_line(sections, i, ref)
        for i, ref in enumerate(vlm_scale_refs, start=1)
    )
    target_index = str(len(vlm_scale_refs) + 1)
    block = _render(
        sections["SCALE_BLOCK"],
        scale_ref_lines=ref_lines,
        target_image_index=target_index,
        asset_id=asset_id,
    )
    return block, target_index


def _build_joints_block(sections: dict[str, str], joints: list[JointSpec], need_has_handle: bool) -> str:
    if not need_has_handle or not joints:
        return sections["JOINTS_BLOCK_NONE"]
    joint_lines = _format_joint_lines(sections["JOINT_LINE"], joints)
    return _render(sections["JOINTS_BLOCK_HAS_HANDLE"], joint_lines=joint_lines)


def _build_door_type_block(sections: dict[str, str], need_door_type: bool) -> str:
    return sections["DOOR_TYPE_BLOCK"] if need_door_type else sections["DOOR_TYPE_NONE"]


def resolve_vlm_prompt_scale_refs(
    category: str,
    scale_method: str,
    calib: dict[str, Any] | None = None,
) -> bool:
    """Whether to include bench scale refs in the VLM prompt (even when scale_method is pixel)."""
    calib = calib or load_calibration_config()
    cfg = calib.get("categories", {}).get(category, {})
    if "vlm_prompt_scale_refs" in cfg:
        return bool(cfg["vlm_prompt_scale_refs"])
    return scale_method in ("vlm_refs", "vlm_hybrid")


def _build_category_hint_block(sections: dict[str, str], category: str) -> str:
    key = f"CATEGORY_HINT_{category.upper()}"
    if key in sections and sections[key].strip():
        return sections[key].strip()
    return sections.get("CATEGORY_HINT_NONE", "").strip()


def build_prompt(
    category: str,
    asset_id: str,
    *,
    vlm_tasks: list[str],
    joints: list[JointSpec] | None = None,
    scale_refs: tuple[ScaleReference, ...] = (),
    vlm_scale_refs: tuple[ScaleReference, ...] | None = None,
    sections: dict[str, str] | None = None,
) -> tuple[str, str]:
    sections = sections or parse_prompt_file()
    joints = list(joints or [])
    active_vlm_refs = vlm_scale_refs if vlm_scale_refs is not None else scale_refs

    need_scale = "scale" in vlm_tasks
    need_has_handle = "has_handle" in vlm_tasks
    need_door_type = "door_type" in vlm_tasks

    scale_block, target_image_index = _build_scale_block(
        sections,
        need_scale=need_scale,
        need_has_handle=need_has_handle,
        need_door_type=need_door_type,
        vlm_scale_refs=active_vlm_refs,
        asset_id=asset_id,
    )

    user_template = _render(sections["USER"], target_image_index=target_image_index)
    user = _render(
        user_template,
        category=category,
        asset_id=asset_id,
        category_hint_block=_build_category_hint_block(sections, category),
        scale_block=scale_block,
        joints_block=_build_joints_block(sections, joints, need_has_handle),
        door_type_block=_build_door_type_block(sections, need_door_type),
    )
    return sections["SYSTEM"], user


def build_prompt_for_asset(
    category: str,
    asset_id: str,
    *,
    dataset_root: Path | None = None,
    vlm_base: dict[str, Any] | None = None,
    sections: dict[str, str] | None = None,
    prompt_path: Path | None = None,
    vlm_base_path: Path | None = None,
    bench_path: Path | None = None,
    calib: dict[str, Any] | None = None,
    use_scale_refs: bool = True,
    vlm_tasks_override: list[str] | None = None,
) -> PromptBundle:
    dataset_root = Path(dataset_root or DEFAULT_DATASET_ROOT)
    vlm_base = vlm_base or load_vlm_base(vlm_base_path)
    sections = sections or parse_prompt_file(prompt_path)

    if category not in vlm_base["categories"]:
        raise KeyError(f"Unknown category: {category}")

    cat_cfg = vlm_base["categories"][category]
    asset_dir = dataset_root / cat_cfg["category_dir"] / asset_id
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    calib = calib or load_calibration_config()
    scale_method = resolve_scale_method(category, calib)
    vlm_tasks = list(vlm_tasks_override or cat_cfg.get("vlm_tasks", ["scale"]))
    scale_refs: tuple[ScaleReference, ...] = ()
    if "scale" in vlm_tasks and use_scale_refs:
        scale_refs = load_scale_references(
            category,
            cat_cfg["category_dir"],
            dataset_root=dataset_root,
            bench_path=bench_path,
            calib=calib,
        )

    if scale_refs and resolve_vlm_prompt_scale_refs(category, scale_method, calib):
        vlm_scale_refs = vlm_scale_refs_for_target(scale_refs, asset_id, leave_one_out=True)
    else:
        vlm_scale_refs = ()

    joints = list_movable_joints(asset_dir)
    system, user = build_prompt(
        category,
        asset_id,
        vlm_tasks=vlm_tasks,
        joints=joints,
        scale_refs=scale_refs,
        vlm_scale_refs=vlm_scale_refs,
        sections=sections,
    )
    image_path = find_category_scale_image(asset_dir, category, calib)

    return PromptBundle(
        system=system,
        user=user,
        category=category,
        asset_id=asset_id,
        asset_dir=asset_dir,
        image_path=image_path,
        vlm_tasks=vlm_tasks,
        joints=tuple(joints),
        scale_refs=scale_refs,
        vlm_scale_refs=vlm_scale_refs,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Preview assembled VLM prompt for one asset.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--vlm-base", type=Path, default=DEFAULT_VLM_BASE_PATH)
    parser.add_argument("--bench-path", type=Path, default=DEFAULT_BENCH_PATH)
    args = parser.parse_args()

    bundle = build_prompt_for_asset(
        args.category,
        args.asset_id,
        dataset_root=args.dataset_root,
        prompt_path=args.prompt,
        vlm_base_path=args.vlm_base,
        bench_path=args.bench_path,
    )
    print(f"target image: {bundle.image_path}")
    if bundle.scale_refs:
        print("scale refs (bench):")
        for i, ref in enumerate(bundle.scale_refs, start=1):
            print(f"  Image {i}: {ref.asset_id} ({ref.l_real_cm} cm) -> {ref.image_path.name}")
    print(f"tasks: {bundle.vlm_tasks}")
    print(f"joints: {len(bundle.joints)}")
    print("--- SYSTEM ---")
    print(bundle.system)
    print("--- USER ---")
    print(bundle.user)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
