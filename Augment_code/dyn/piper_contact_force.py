"""Piper fingertip contact-force sensing and CSV logging for Isaac keyboard collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.schemas import activate_contact_sensors


@dataclass(frozen=True)
class ContactForceTaskConfig:
    """Task-specific rigid bodies whose contacts should be measured."""

    contact_prim_paths: tuple[str, ...]


# Only the laptop task is configured currently. The paths were verified from
# tasks/open_laptop/data/scene.usd and exclude ground and self-contact.
TASK_CONTACT_FORCE_CONFIGS: dict[str, ContactForceTaskConfig] = {
    "close_laptop_lid": ContactForceTaskConfig(
        contact_prim_paths=(
            "/World/generated/base",
            "/World/generated/link_1",
        )
    ),
    "adjust_the_faucet": ContactForceTaskConfig(
        contact_prim_paths=(
            "/World/mobility_isaac/base",
            "/World/mobility_isaac/link_0",
        )
    ),
    "adjust_the_monitor": ContactForceTaskConfig(
        contact_prim_paths=(
            "/World/generated/base",
            "/World/generated/link_1",
        )
    ),
}


def create_piper_finger_force_sensors(task_id: str, robot_prim_path: str, stage) -> tuple[
    dict[str, ContactSensor], ContactForceTaskConfig
]:
    """Create filtered world-frame force sensors for Piper's two gripper fingers."""
    try:
        task_config = TASK_CONTACT_FORCE_CONFIGS[task_id]
    except KeyError as exc:
        raise ValueError(
            f"Task '{task_id}' has no contact-force configuration in Augment_code/dyn."
        ) from exc

    robot_prim_root = robot_prim_path.rsplit("/", 1)[0]
    finger_sensor_paths = {
        "link7": f"{robot_prim_root}/link7",
        "link8": f"{robot_prim_root}/link8",
    }
    missing_force_prims = [
        prim_path
        for prim_path in (*finger_sensor_paths.values(), *task_config.contact_prim_paths)
        if not stage.GetPrimAtPath(prim_path).IsValid()
    ]
    if missing_force_prims:
        raise RuntimeError(
            "Cannot initialize force logging because these required rigid-body prims are missing: "
            + ", ".join(missing_force_prims)
        )

    for prim_path in finger_sensor_paths.values():
        activate_contact_sensors(prim_path)

    sensors = {
        finger_name: ContactSensor(
            ContactSensorCfg(
                prim_path=prim_path,
                update_period=0.0,
                history_length=0,
                filter_prim_paths_expr=list(task_config.contact_prim_paths),
            )
        )
        for finger_name, prim_path in finger_sensor_paths.items()
    }
    print(
        "[INFO] Force sensors: "
        f"fingers={list(finger_sensor_paths.values())}, "
        f"task_object={list(task_config.contact_prim_paths)}"
    )
    return sensors, task_config


def update_finger_force_sensors(finger_force_sensors: Mapping[str, ContactSensor], sim_dt: float) -> None:
    """Refresh each sensor after its corresponding PhysX simulation step."""
    for finger_sensor in finger_force_sensors.values():
        finger_sensor.update(sim_dt, force_recompute=True)


def read_piper_finger_force_sample(
    finger_force_sensors: Mapping[str, ContactSensor],
) -> tuple[list[float], list[float], list[float], bool]:
    """Read world-frame forces in N and return link7, link8, total, and contact flag."""
    link7_force_n = finger_force_sensors["link7"].data.net_forces_w[0, 0].detach().cpu().tolist()
    link8_force_n = finger_force_sensors["link8"].data.net_forces_w[0, 0].detach().cpu().tolist()
    total_force_n = [
        float(link7_force_n[axis]) + float(link8_force_n[axis])
        for axis in range(3)
    ]
    in_contact = sum(component * component for component in total_force_n) ** 0.5 > 1.0e-4
    return link7_force_n, link8_force_n, total_force_n, in_contact
