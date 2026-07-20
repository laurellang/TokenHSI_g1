#!/usr/bin/env python3
"""View AMP demo and policy rollout together as URDF meshes in viser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from visualize_g1_smoke_rollout import (
    load_mesh,
    make_transform,
    make_visual_pose,
    parse_urdf_visuals,
    tint_mesh,
)


def load_rollout(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "body_names": tuple(str(name) for name in data["body_names"].tolist()),
            "body_pos": np.asarray(data["body_pos"], dtype=np.float32),
            "body_quat_xyzw": np.asarray(data["body_quat_xyzw"], dtype=np.float32),
            "dt": float(np.asarray(data["dt"]).reshape(())),
        }


def align_body_pos(body_pos: np.ndarray, offset: Sequence[float]) -> np.ndarray:
    aligned = np.array(body_pos, dtype=np.float32, copy=True)
    root_xy = aligned[0, 0, 0:2].copy()
    aligned[..., 0:2] -= root_xy
    aligned += np.asarray(offset, dtype=np.float32).reshape(1, 1, 3)
    return aligned


def colored_mesh(mesh_path: Path, scale: np.ndarray, color: Sequence[int]):
    mesh = load_mesh(mesh_path, scale)
    return tint_mesh(mesh, [int(color[0]), int(color[1]), int(color[2]), 210])


def add_actor_meshes(server, prefix: str, rollout, body_pos: np.ndarray, urdf_path: Path, color: Sequence[int]):
    visuals = parse_urdf_visuals(urdf_path)
    body_to_index = {name: index for index, name in enumerate(rollout["body_names"])}
    mesh_cache: Dict[Tuple[Path, Tuple[float, float, float]], object] = {}
    handles = []
    for visual_index, visual in enumerate(visuals):
        body_index = body_to_index.get(visual.link_name)
        if body_index is None:
            continue
        cache_key = (visual.mesh_path, tuple(float(value) for value in visual.scale))
        mesh = mesh_cache.get(cache_key)
        if mesh is None:
            mesh = colored_mesh(visual.mesh_path, visual.scale, color)
            mesh_cache[cache_key] = mesh
        handle = server.scene.add_mesh_trimesh(f"/{prefix}/{visual.link_name}/visual_{visual_index}", mesh)
        visual_transform = make_transform(visual.origin_xyz, visual.origin_rpy)
        inertial_transform = make_transform(visual.inertial_xyz, visual.inertial_rpy)
        handles.append((body_index, visual_transform, inertial_transform, handle, body_pos, rollout["body_quat_xyzw"]))
    return handles


def parse_color(text: str):
    parts = [int(part) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be R,G,B")
    return tuple(int(np.clip(part, 0, 255)) for part in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default="/tmp/g1_amp_demo_loco_motion0.npz")
    parser.add_argument("--policy", default="/tmp/g1_loco_reward_latest_1000_rollout.npz")
    parser.add_argument("--urdf", default="tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--play-dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--ground-size", type=float, default=4.0)
    parser.add_argument("--demo-offset", type=float, nargs=3, default=[0.0, 0.8, 0.0])
    parser.add_argument("--policy-offset", type=float, nargs=3, default=[0.0, -0.8, 0.0])
    parser.add_argument("--demo-color", type=parse_color, default=(40, 190, 95))
    parser.add_argument("--policy-color", type=parse_color, default=(235, 135, 35))
    args = parser.parse_args()

    import viser

    demo = load_rollout(Path(args.demo).resolve())
    policy = load_rollout(Path(args.policy).resolve())
    urdf_path = Path(args.urdf).resolve()
    demo_pos = align_body_pos(demo["body_pos"], args.demo_offset)
    policy_pos = align_body_pos(policy["body_pos"], args.policy_offset)
    num_frames = min(demo_pos.shape[0], policy_pos.shape[0])

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/ground", width=args.ground_size, height=args.ground_size)
    handles = []
    handles.extend(add_actor_meshes(server, "amp_demo", demo, demo_pos, urdf_path, args.demo_color))
    handles.extend(add_actor_meshes(server, "policy", policy, policy_pos, urdf_path, args.policy_color))

    status = server.gui.add_text("frame", initial_value="")
    playing = server.gui.add_checkbox("play", initial_value=True)
    frame_slider = server.gui.add_slider("frame", min=0, max=num_frames - 1, step=1, initial_value=0)

    def set_frame(frame_index: int) -> None:
        frame_index = int(np.clip(frame_index, 0, num_frames - 1))
        for body_index, visual_transform, inertial_transform, handle, body_pos, body_quat in handles:
            pos, quat = make_visual_pose(
                body_pos[frame_index, body_index],
                body_quat[frame_index, body_index],
                visual_transform,
                inertial_transform=inertial_transform,
            )
            handle.position = pos
            handle.wxyz = quat
        status.value = f"{frame_index}/{num_frames - 1}  green=AMP demo  orange=policy"

    @frame_slider.on_update
    def _(_) -> None:
        set_frame(int(frame_slider.value))

    set_frame(0)
    print(f"Viser demo-vs-policy mesh running on http://{args.host}:{args.port}")
    next_frame_time = time.time()
    while True:
        if playing.value:
            now = time.time()
            if now >= next_frame_time:
                frame_slider.value = int((int(frame_slider.value) + 1) % num_frames)
                set_frame(int(frame_slider.value))
                next_frame_time = now + max(float(args.play_dt), 1.0e-3)
        time.sleep(0.005)


if __name__ == "__main__":
    raise SystemExit(main())
