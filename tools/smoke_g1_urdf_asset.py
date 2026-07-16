#!/usr/bin/env python3
"""Smoke-test an IsaacGym asset and print its body/DOF contract.

Example:
    conda activate tokenhsi_g1
    python tools/smoke_g1_urdf_asset.py \
        --asset-root assets/g1_with_brainco_hand \
        --asset-file g1_29dof_mode_15_brainco_hand.urdf \
        --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacgym import gymapi


def _drive_mode_name(mode: int) -> str:
    names = {
        gymapi.DOF_MODE_NONE: "none",
        gymapi.DOF_MODE_POS: "pos",
        gymapi.DOF_MODE_VEL: "vel",
        gymapi.DOF_MODE_EFFORT: "effort",
    }
    return names.get(int(mode), str(int(mode)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", default="assets/g1_with_brainco_hand")
    parser.add_argument("--asset-file", default="g1_29dof_mode_15_brainco_hand.urdf")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fix-base-link", action="store_true")
    parser.add_argument("--collapse-fixed-joints", action="store_true")
    parser.add_argument("--disable-gravity", action="store_true")
    parser.add_argument("--no-actor", action="store_true", help="Only load the asset; do not instantiate it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_root = Path(args.asset_root).resolve()
    asset_file = args.asset_file
    asset_path = asset_root / asset_file
    if not asset_path.exists():
        raise FileNotFoundError(asset_path)

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create IsaacGym sim")

    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_options.fix_base_link = args.fix_base_link
    asset_options.collapse_fixed_joints = args.collapse_fixed_joints
    asset_options.disable_gravity = args.disable_gravity

    print("ASSET")
    print(f"  root: {asset_root}")
    print(f"  file: {asset_file}")
    print(f"  exists: {asset_path.exists()}")
    print("OPTIONS")
    print(f"  fix_base_link: {asset_options.fix_base_link}")
    print(f"  collapse_fixed_joints: {asset_options.collapse_fixed_joints}")
    print(f"  default_dof_drive_mode: {_drive_mode_name(asset_options.default_dof_drive_mode)}")

    asset = gym.load_asset(sim, str(asset_root), asset_file, asset_options)
    if asset is None:
        raise RuntimeError("gym.load_asset returned None")

    body_count = gym.get_asset_rigid_body_count(asset)
    dof_count = gym.get_asset_dof_count(asset)
    joint_count = gym.get_asset_joint_count(asset)
    actuator_count = gym.get_asset_actuator_count(asset)
    tendon_count = gym.get_asset_tendon_count(asset)

    print("COUNTS")
    print(f"  bodies: {body_count}")
    print(f"  joints: {joint_count}")
    print(f"  dofs: {dof_count}")
    print(f"  actuators: {actuator_count}")
    print(f"  tendons: {tendon_count}")

    body_names = gym.get_asset_rigid_body_names(asset)
    dof_names = gym.get_asset_dof_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)

    print("BODIES")
    for i, name in enumerate(body_names):
        print(f"  {i:03d} {name}")

    print("DOFS")
    for i, name in enumerate(dof_names):
        lower = float(dof_props["lower"][i])
        upper = float(dof_props["upper"][i])
        effort = float(dof_props["effort"][i])
        velocity = float(dof_props["velocity"][i])
        drive_mode = _drive_mode_name(dof_props["driveMode"][i])
        stiffness = float(dof_props["stiffness"][i])
        damping = float(dof_props["damping"][i])
        armature = float(dof_props["armature"][i])
        print(
            f"  {i:03d} {name} "
            f"range=[{lower:.7g}, {upper:.7g}] "
            f"effort={effort:.7g} velocity={velocity:.7g} "
            f"drive={drive_mode} stiffness={stiffness:.7g} "
            f"damping={damping:.7g} armature={armature:.7g}"
        )

    if not args.no_actor:
        env = gym.create_env(
            sim,
            gymapi.Vec3(-1.0, -1.0, 0.0),
            gymapi.Vec3(1.0, 1.0, 1.0),
            1,
        )
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.8)
        actor = gym.create_actor(env, asset, pose, "g1_braincohand", 0, 1)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        actor_body_count = gym.get_actor_rigid_body_count(env, actor)
        actor_dof_count = gym.get_actor_dof_count(env, actor)
        print("ACTOR")
        print(f"  bodies: {actor_body_count}")
        print(f"  dofs: {actor_dof_count}")

    gym.destroy_sim(sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
