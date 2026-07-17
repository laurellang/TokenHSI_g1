#!/usr/bin/env python3
"""Record an IsaacGym G1 smoke-test trajectory and replay it as URDF meshes in viser.

Examples:
    python tools/visualize_g1_smoke_rollout.py record --headless --steps 300
    python tools/visualize_g1_smoke_rollout.py view --rollout /tmp/g1_smoke_rollout.npz --port 8080
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RolloutData:
    body_names: Tuple[str, ...]
    body_pos: np.ndarray
    body_quat_xyzw: np.ndarray
    dt: float


@dataclass(frozen=True)
class VisualSpec:
    link_name: str
    mesh_path: Path
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    scale: np.ndarray
    inertial_xyz: np.ndarray
    inertial_rpy: np.ndarray


def parse_vec(text: Optional[str], default: Iterable[float]) -> np.ndarray:
    if not text:
        return np.array(tuple(default), dtype=np.float64)
    return np.array([float(part) for part in text.split()], dtype=np.float64)


def rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return rot_z(float(rpy[2])) @ rot_y(float(rpy[1])) @ rot_x(float(rpy[0]))


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def make_visual_pose(
    body_pos: np.ndarray,
    body_quat_xyzw: np.ndarray,
    visual_transform: np.ndarray,
    inertial_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    body_transform = np.eye(4, dtype=np.float64)
    body_transform[:3, :3] = quat_xyzw_to_matrix(body_quat_xyzw)
    body_transform[:3, 3] = np.asarray(body_pos, dtype=np.float64)
    if inertial_transform is not None:
        link_transform = body_transform @ np.linalg.inv(inertial_transform)
    else:
        link_transform = body_transform
    world_visual = link_transform @ visual_transform
    return world_visual[:3, 3], quat_wxyz_from_matrix(world_visual[:3, :3])


def load_rollout(path: Path) -> RolloutData:
    with np.load(path, allow_pickle=False) as data:
        body_names = tuple(str(name) for name in data["body_names"].tolist())
        body_pos = np.asarray(data["body_pos"], dtype=np.float64)
        body_quat_xyzw = np.asarray(data["body_quat_xyzw"], dtype=np.float64)
        dt = float(np.asarray(data["dt"]).reshape(()))

    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(f"body_pos must have shape [frames, bodies, 3], got {body_pos.shape}")
    if body_quat_xyzw.ndim != 3 or body_quat_xyzw.shape[-1] != 4:
        raise ValueError(f"body_quat_xyzw must have shape [frames, bodies, 4], got {body_quat_xyzw.shape}")
    if body_pos.shape[:2] != body_quat_xyzw.shape[:2]:
        raise ValueError(f"body_pos and body_quat_xyzw frame/body dimensions differ: {body_pos.shape}, {body_quat_xyzw.shape}")
    if body_pos.shape[1] != len(body_names):
        raise ValueError(f"body_names has {len(body_names)} entries but rollout has {body_pos.shape[1]} bodies")
    return RolloutData(body_names=body_names, body_pos=body_pos, body_quat_xyzw=body_quat_xyzw, dt=dt)


def extract_rigid_body_pose_arrays(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pose = state["pose"]
    pos = np.stack(
        [pose["p"]["x"], pose["p"]["y"], pose["p"]["z"]],
        axis=-1,
    ).astype(np.float64, copy=False)
    quat_xyzw = np.stack(
        [pose["r"]["x"], pose["r"]["y"], pose["r"]["z"], pose["r"]["w"]],
        axis=-1,
    ).astype(np.float64, copy=False)
    return pos, quat_xyzw


def resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
        parts = filename.split("/", 1)
        filename = parts[1] if len(parts) == 2 else parts[0]
    return (urdf_path.parent / filename).resolve()


def parse_urdf_visuals(urdf_path: Path) -> List[VisualSpec]:
    root = ET.parse(urdf_path).getroot()
    visuals: List[VisualSpec] = []
    for link_elem in root.findall("link"):
        link_name = link_elem.attrib["name"]
        inertial_elem = link_elem.find("inertial")
        inertial_origin_elem = inertial_elem.find("origin") if inertial_elem is not None else None
        inertial_xyz = parse_vec(inertial_origin_elem.attrib.get("xyz") if inertial_origin_elem is not None else None, [0.0, 0.0, 0.0])
        inertial_rpy = parse_vec(inertial_origin_elem.attrib.get("rpy") if inertial_origin_elem is not None else None, [0.0, 0.0, 0.0])
        for visual_elem in link_elem.findall("visual"):
            geometry_elem = visual_elem.find("geometry")
            mesh_elem = geometry_elem.find("mesh") if geometry_elem is not None else None
            if mesh_elem is None or not mesh_elem.attrib.get("filename"):
                continue
            origin_elem = visual_elem.find("origin")
            visuals.append(
                VisualSpec(
                    link_name=link_name,
                    mesh_path=resolve_mesh_path(urdf_path, mesh_elem.attrib["filename"]),
                    origin_xyz=parse_vec(origin_elem.attrib.get("xyz") if origin_elem is not None else None, [0.0, 0.0, 0.0]),
                    origin_rpy=parse_vec(origin_elem.attrib.get("rpy") if origin_elem is not None else None, [0.0, 0.0, 0.0]),
                    scale=parse_vec(mesh_elem.attrib.get("scale"), [1.0, 1.0, 1.0]),
                    inertial_xyz=inertial_xyz,
                    inertial_rpy=inertial_rpy,
                )
            )
    return visuals


def load_mesh(mesh_path: Path, scale: np.ndarray):
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not np.allclose(scale, np.ones(3)):
        mesh = mesh.copy()
        mesh.apply_scale(scale)
    return mesh


def tint_mesh(mesh, rgba: Sequence[int]):
    tinted = mesh.copy()
    tinted.visual.vertex_colors = np.array(rgba, dtype=np.uint8)
    return tinted


def record_rollout(args: argparse.Namespace) -> int:
    from isaacgym import gymapi

    asset_root = Path(args.asset_root).resolve()
    asset_file = args.asset_file
    asset_path = asset_root / asset_file
    if not asset_path.exists():
        raise FileNotFoundError(asset_path)

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = float(args.dt)
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0 if args.disable_gravity else -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0

    sim = gym.create_sim(args.compute_device, args.graphics_device, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create IsaacGym sim")

    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    asset_options.fix_base_link = bool(args.fix_base_link)
    asset_options.collapse_fixed_joints = bool(args.collapse_fixed_joints)
    asset_options.disable_gravity = bool(args.disable_gravity)

    asset = gym.load_asset(sim, str(asset_root), asset_file, asset_options)
    if asset is None:
        raise RuntimeError("gym.load_asset returned None")
    body_names = tuple(gym.get_asset_rigid_body_names(asset))

    env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(args.root_x), float(args.root_y), float(args.root_z))
    actor = gym.create_actor(env, asset, pose, "g1_smoke", 0, 1)
    if actor < 0:
        raise RuntimeError("Failed to create G1 actor")

    frames_pos = []
    frames_quat = []
    for _step in range(int(args.steps)):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        body_state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        pos, quat = extract_rigid_body_pose_arrays(body_state[: len(body_names)])
        frames_pos.append(pos.copy())
        frames_quat.append(quat.copy())

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        body_names=np.array(body_names),
        body_pos=np.stack(frames_pos, axis=0).astype(np.float32),
        body_quat_xyzw=np.stack(frames_quat, axis=0).astype(np.float32),
        dt=np.array(float(args.dt), dtype=np.float32),
        asset_root=np.array(str(asset_root)),
        asset_file=np.array(asset_file),
    )
    gym.destroy_sim(sim)
    print(f"saved {len(frames_pos)} frames, {len(body_names)} bodies -> {output}")
    return 0


def view_rollout(args: argparse.Namespace) -> int:
    import viser

    rollout = load_rollout(Path(args.rollout).resolve())
    urdf_path = Path(args.urdf).resolve()
    visuals = parse_urdf_visuals(urdf_path)
    body_to_index = {name: index for index, name in enumerate(rollout.body_names)}

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/ground", width=args.ground_size, height=args.ground_size)

    mesh_cache: Dict[Tuple[Path, Tuple[float, float, float], str], object] = {}
    handles = []
    for visual_index, visual in enumerate(visuals):
        body_index = body_to_index.get(visual.link_name)
        if body_index is None:
            continue
        cache_key = (visual.mesh_path, tuple(float(value) for value in visual.scale), visual.link_name)
        mesh = mesh_cache.get(cache_key)
        if mesh is None:
            mesh = load_mesh(visual.mesh_path, visual.scale)
            if visual.link_name == args.head_body:
                mesh = tint_mesh(mesh, [230, 40, 40, 255])
            elif visual.link_name == args.torso_body:
                mesh = tint_mesh(mesh, [40, 110, 230, 255])
            else:
                mesh = tint_mesh(mesh, [210, 210, 215, 255])
            mesh_cache[cache_key] = mesh
        handle = server.scene.add_mesh_trimesh(f"/smoke/{visual.link_name}/visual_{visual_index}", mesh)
        visual_transform = make_transform(visual.origin_xyz, visual.origin_rpy)
        inertial_transform = None if args.no_inertial_correction else make_transform(visual.inertial_xyz, visual.inertial_rpy)
        handles.append((body_index, visual_transform, inertial_transform, handle))

    head_index = body_to_index.get(args.head_body)
    torso_index = body_to_index.get(args.torso_body)
    line_handle = None
    if head_index is not None and torso_index is not None:
        line_handle = server.scene.add_line_segments(
            "/debug/torso_to_head",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.array([[[40, 110, 230], [230, 40, 40]]], dtype=np.uint8),
            line_width=3.0,
        )

    status = server.gui.add_text("frame", initial_value="")
    playing = server.gui.add_checkbox("play", initial_value=True)
    frame_slider = server.gui.add_slider("frame", min=0, max=rollout.body_pos.shape[0] - 1, step=1, initial_value=0)

    def set_frame(frame_index: int) -> None:
        frame_index = int(np.clip(frame_index, 0, rollout.body_pos.shape[0] - 1))
        for body_index, visual_transform, inertial_transform, handle in handles:
            pos, quat = make_visual_pose(
                rollout.body_pos[frame_index, body_index],
                rollout.body_quat_xyzw[frame_index, body_index],
                visual_transform,
                inertial_transform=inertial_transform,
            )
            handle.position = pos
            handle.wxyz = quat
        if line_handle is not None and head_index is not None and torso_index is not None:
            head = rollout.body_pos[frame_index, head_index]
            torso = rollout.body_pos[frame_index, torso_index]
            line_handle.points = np.array([[torso, head]], dtype=np.float32)
            dist = float(np.linalg.norm(head - torso))
            status.value = f"{frame_index}/{rollout.body_pos.shape[0] - 1}  {args.head_body}-{args.torso_body}: {dist:.6f} m"
        else:
            status.value = f"{frame_index}/{rollout.body_pos.shape[0] - 1}"

    @frame_slider.on_update
    def _(_event) -> None:
        set_frame(int(frame_slider.value))

    set_frame(0)
    print(f"Viser mesh rollout running for {Path(args.rollout).resolve()}")
    print(f"Open http://{args.host}:{args.port}")
    next_frame_time = time.time()
    while True:
        if playing.value:
            now = time.time()
            if now >= next_frame_time:
                frame_slider.value = int((int(frame_slider.value) + 1) % rollout.body_pos.shape[0])
                set_frame(int(frame_slider.value))
                next_frame_time = now + max(float(args.play_dt), 1e-3)
        time.sleep(0.005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Run IsaacGym smoke test and save rigid-body states.")
    record.add_argument("--asset-root", default="tokenhsi/data/assets")
    record.add_argument("--asset-file", default="mjcf/g1_dex3_ori.urdf")
    record.add_argument("--output", default="/tmp/g1_smoke_rollout.npz")
    record.add_argument("--steps", type=int, default=300)
    record.add_argument("--dt", type=float, default=1.0 / 60.0)
    record.add_argument("--root-x", type=float, default=0.0)
    record.add_argument("--root-y", type=float, default=0.0)
    record.add_argument("--root-z", type=float, default=0.8)
    record.add_argument("--compute-device", type=int, default=0)
    record.add_argument("--graphics-device", type=int, default=0)
    record.add_argument("--fix-base-link", action="store_true")
    record.add_argument("--collapse-fixed-joints", action="store_true")
    record.add_argument("--disable-gravity", action="store_true")
    record.set_defaults(func=record_rollout)

    view = subparsers.add_parser("view", help="Replay a saved smoke rollout using URDF visual meshes in viser.")
    view.add_argument("--rollout", default="/tmp/g1_smoke_rollout.npz")
    view.add_argument("--urdf", default="tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--ground-size", type=float, default=4.0)
    view.add_argument("--play-dt", type=float, default=1.0 / 30.0)
    view.add_argument("--head-body", default="head_link")
    view.add_argument("--torso-body", default="torso_link")
    view.add_argument("--no-inertial-correction", action="store_true", help="Treat saved rigid body poses as URDF link frames instead of IsaacGym body/inertial frames.")
    view.set_defaults(func=view_rollout)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
