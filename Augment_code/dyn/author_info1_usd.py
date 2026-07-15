#!/usr/bin/env python3
"""Persist info1.json values into an already converted layered USD asset."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("asset_dir", type=Path)
parser.add_argument("--output-name", default="fitr_info1.usd")
parser.add_argument(
    "--rotation-x-deg",
    type=float,
    help="Set the root prim's X-axis rotation in degrees.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def open_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {path}")
    stage.SetEditTarget(stage.GetRootLayer())
    return stage


def child_prim(stage: Usd.Stage, relative_path: str, purpose: str) -> Usd.Prim:
    root = stage.GetDefaultPrim()
    prim = stage.GetPrimAtPath(root.GetPath().AppendPath(relative_path))
    if not prim:
        raise ValueError(f"Missing {purpose} prim at {root.GetPath()}/{relative_path}")
    return prim


def urdf_root(asset_dir: Path) -> ET.Element:
    urdf_path = next(
        (path for path in (asset_dir / "mobility.urdf", asset_dir / "test.urdf") if path.is_file()),
        None,
    )
    if urdf_path is None:
        raise FileNotFoundError(f"No URDF found in {asset_dir}")
    return ET.parse(urdf_path).getroot()


def visual_link_names(robot: ET.Element, semantic_name: str) -> set[str]:
    return {
        link.get("name")
        for link in robot.findall("link")
        if any(
            visual.get("name") in {semantic_name, f"{semantic_name}-"}
            or str(visual.get("name", "")).startswith(f"{semantic_name}-")
            for visual in link.findall("visual")
        )
    }


def resolve_joint_name(asset_dir: Path, data: dict) -> str:
    robot = urdf_root(asset_dir)

    expected_type = str(data["motion_type"]).lower()
    valid_types = {"revolute", "continuous"} if expected_type == "revolute" else {expected_type}
    named_joint = [
        joint
        for joint in robot.findall("joint")
        if joint.get("name") == str(data["joint_name"]) and joint.get("type") in valid_types
    ]
    if len(named_joint) == 1:
        return named_joint[0].get("name")

    candidates = [
        joint
        for joint in robot.findall("joint")
        if joint.get("type") in valid_types
        and joint.find("child").get("link") == str(data["link_name"])
    ]
    semantic_links = visual_link_names(robot, str(data["link_name"]))
    if len(candidates) != 1 and semantic_links:
        candidates = [
            joint
            for joint in robot.findall("joint")
            if joint.get("type") in valid_types
            and joint.find("child").get("link") in semantic_links
        ]
    if len(candidates) != 1 and semantic_links:
        parent_joint = {joint.find("child").get("link"): joint for joint in robot.findall("joint")}
        ancestor_candidates = []
        for link_name in semantic_links:
            visited = set()
            while link_name in parent_joint and link_name not in visited:
                visited.add(link_name)
                joint = parent_joint[link_name]
                if joint.get("type") in valid_types:
                    ancestor_candidates.append(joint)
                    break
                link_name = joint.find("parent").get("link")
        if len(ancestor_candidates) == 1:
            candidates = ancestor_candidates
    if len(candidates) != 1:
        index = re.fullmatch(r"joint_(\d+)", str(data["joint_name"]))
        if index and len(candidates) > 1:
            candidate_index = int(index.group(1)) - len(candidates)
            if 0 <= candidate_index < len(candidates):
                return sorted(candidates, key=lambda joint: joint.get("name"))[candidate_index].get("name")
        movable_joints = [
            joint for joint in robot.findall("joint") if joint.get("type") in valid_types
        ]
        if index and int(index.group(1)) < len(movable_joints):
            return movable_joints[int(index.group(1))].get("name")
        names = [joint.get("name") for joint in candidates]
        raise ValueError(
            f"Cannot map info1 {data['joint_name']} to one {expected_type} joint "
            f"for link {data['link_name']}; candidates={names}"
        )
    return candidates[0].get("name")


def resolve_link_name(asset_dir: Path, data: dict) -> str:
    semantic_name = str(data["link_name"])
    visual_links = visual_link_names(urdf_root(asset_dir), semantic_name)
    if len(visual_links) == 1:
        return visual_links.pop()

    joint_name = resolve_joint_name(asset_dir, data)
    joint = next(
        joint for joint in urdf_root(asset_dir).findall("joint") if joint.get("name") == joint_name
    )
    return joint.find("child").get("link")


def author_root_and_affordances(stage: Usd.Stage, info: dict) -> tuple[Usd.Prim, float, list]:
    scale_data = info["scale"]
    scale_m = float(scale_data["scale"]) / 100.0
    if scale_m <= 0:
        raise ValueError(f"Invalid info1 scale: {scale_data['scale']}")

    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("Base USD has no default prim")
    xformable = UsdGeom.Xformable(root)
    scale_ops = [
        op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    scale_op = scale_ops[0] if scale_ops else xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scale_m))
    if args.rotation_x_deg is not None:
        rotation_ops = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateX
        ]
        rotation_op = rotation_ops[0] if rotation_ops else xformable.AddRotateXOp()
        rotation_op.Set(args.rotation_x_deg)

    root.CreateAttribute("fitr:sourceScaleCmPerUnit", Sdf.ValueTypeNames.Double).Set(
        float(scale_data["scale"])
    )
    root.CreateAttribute("fitr:realSizeCm", Sdf.ValueTypeNames.Double).Set(
        float(scale_data["L_real_cm"])
    )
    root.CreateAttribute("fitr:assetId", Sdf.ValueTypeNames.String).Set(str(info["asset_id"]))
    root.CreateAttribute("fitr:category", Sdf.ValueTypeNames.String).Set(str(info["category"]))

    link_paths = []
    for affordance in info.get("affordances", []):
        position = affordance.get("position_xyz")
        axis = affordance.get("contact_axis_xyz")
        if not (
            isinstance(position, list)
            and len(position) == 3
            and isinstance(axis, list)
            and len(axis) == 3
            and all(isinstance(value, (int, float)) for value in position + axis)
        ):
            continue
        link_name = resolve_link_name(args.asset_dir, affordance)
        link = child_prim(stage, link_name, "affordance")
        link.CreateAttribute("fitr:affordance:position", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(*(float(value) for value in position))
        )
        link.CreateAttribute("fitr:affordance:contactAxis", Sdf.ValueTypeNames.Double3).Set(
            Gf.Vec3d(*(float(value) for value in axis))
        )
        link.CreateAttribute("fitr:affordance:jointName", Sdf.ValueTypeNames.String).Set(
            str(affordance["joint_name"])
        )
        link.CreateAttribute("fitr:affordance:hasHandle", Sdf.ValueTypeNames.Bool).Set(
            bool(affordance.get("has_handle", False))
        )
        link_paths.append(link.GetPath())
    return root, scale_m, link_paths


def author_joint(stage: Usd.Stage, data: dict, joint_name: str) -> tuple[Usd.Prim, str]:
    prim = child_prim(stage, f"joints/{joint_name}", "joint dynamics")
    motion_type = str(data["motion_type"]).lower()
    if motion_type in {"revolute", "continuous"}:
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise TypeError(f"{prim.GetPath()} is not a revolute USD joint")
        drive_name = "angular"
    elif motion_type == "prismatic":
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            raise TypeError(f"{prim.GetPath()} is not a prismatic USD joint")
        drive_name = "linear"
    else:
        raise ValueError(f"Unsupported motion_type: {motion_type}")

    drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    drive.CreateTypeAttr("force")
    drive.CreateDampingAttr(float(data["damping"]))
    drive.CreateStiffnessAttr(float(data["stiffness"]))
    drive.CreateMaxForceAttr(float(data["effort_limit"]))
    return prim, drive_name


def main() -> None:
    asset_dir = args.asset_dir.expanduser().resolve()
    output_path = asset_dir / args.output_name
    base_path = asset_dir / "configuration" / f"{output_path.stem}_base.usd"
    physics_path = asset_dir / "configuration" / f"{output_path.stem}_physics.usd"
    with (asset_dir / "info1.json").open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    base_stage = open_stage(base_path)
    root, scale_m, link_paths = author_root_and_affordances(base_stage, info)
    root_path = root.GetPath()
    base_stage.GetRootLayer().Save()

    physics_stage = open_stage(physics_path)
    joints = []
    for data in info["joints"]:
        joint_name = resolve_joint_name(asset_dir, data)
        prim, drive_name = author_joint(physics_stage, data, joint_name)
        joints.append((prim.GetPath(), drive_name, data))
    physics_stage.GetRootLayer().Save()

    # Validate the final composed file and require the expected owning layers.
    stage = open_stage(output_path)
    check_root = stage.GetPrimAtPath(root_path)
    scale_ops = [
        op
        for op in UsdGeom.Xformable(check_root).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if len(scale_ops) != 1 or any(abs(float(v) - scale_m) > 1e-9 for v in scale_ops[0].Get()):
        raise RuntimeError(f"Persisted scale verification failed: {scale_ops}")
    scale_stack = scale_ops[0].GetAttr().GetPropertyStack()
    if Path(scale_stack[0].layer.identifier).resolve() != base_path.resolve():
        raise RuntimeError(f"Scale is not owned by {base_path}")
    if args.rotation_x_deg is not None:
        rotation_ops = [
            op
            for op in UsdGeom.Xformable(check_root).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateX
        ]
        if len(rotation_ops) != 1 or abs(float(rotation_ops[0].Get()) - args.rotation_x_deg) > 1e-9:
            raise RuntimeError(f"Persisted X rotation verification failed: {rotation_ops}")
        rotation_stack = rotation_ops[0].GetAttr().GetPropertyStack()
        if Path(rotation_stack[0].layer.identifier).resolve() != base_path.resolve():
            raise RuntimeError(f"X rotation is not owned by {base_path}")

    verified_joints = []
    for joint_path, drive_name, expected in joints:
        prim = stage.GetPrimAtPath(joint_path)
        drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
        values = {
            "joint": str(joint_path),
            "drive": drive_name,
            "damping": float(drive.GetDampingAttr().Get()),
            "stiffness": float(drive.GetStiffnessAttr().Get()),
            "max_force": float(drive.GetMaxForceAttr().Get()),
        }
        for key, source_key in (
            ("damping", "damping"),
            ("stiffness", "stiffness"),
            ("max_force", "effort_limit"),
        ):
            if abs(values[key] - float(expected[source_key])) > 1e-4:
                raise RuntimeError(f"{joint_path} {key} verification failed: {values[key]}")
        stack = drive.GetDampingAttr().GetPropertyStack()
        if Path(stack[0].layer.identifier).resolve() != physics_path.resolve():
            raise RuntimeError(f"{joint_path} dynamics are not owned by {physics_path}")
        verified_joints.append(values)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "scale": scale_m,
                "scale_layer": str(base_path),
                "joints": verified_joints,
                "dynamics_layer": str(physics_path),
                "affordance_links": [str(path) for path in link_paths],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
