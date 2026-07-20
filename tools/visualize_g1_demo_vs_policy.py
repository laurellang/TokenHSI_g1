#!/usr/bin/env python3
"""Export AMP demo motion and compare it with a policy rollout in viser."""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def load_rollout(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "body_names": tuple(str(name) for name in data["body_names"].tolist()),
            "body_pos": np.asarray(data["body_pos"], dtype=np.float32),
            "body_quat_xyzw": np.asarray(data["body_quat_xyzw"], dtype=np.float32),
            "dt": float(np.asarray(data["dt"]).reshape(())),
        }


def parse_urdf_edges(urdf_path: Path, body_names: Sequence[str]) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    body_to_index = {name: index for index, name in enumerate(body_names)}
    edges: List[Tuple[int, int]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        if parent_name in body_to_index and child_name in body_to_index:
            edges.append((body_to_index[parent_name], body_to_index[child_name]))
    if not edges:
        raise ValueError(f"No URDF edges matched rollout body names from {urdf_path}")
    return np.asarray(edges, dtype=np.int64)


def align_body_pos(body_pos: np.ndarray, offset: Sequence[float]) -> np.ndarray:
    aligned = np.array(body_pos, dtype=np.float32, copy=True)
    root_xy = aligned[0, 0, 0:2].copy()
    aligned[..., 0:2] -= root_xy
    aligned += np.asarray(offset, dtype=np.float32).reshape(1, 1, 3)
    return aligned


def export_demo(args: argparse.Namespace) -> int:
    import torch
    from utils.gmr_robot_motion_lib import GMRRobotMotionLib

    motion_lib = GMRRobotMotionLib(
        motion_file=args.motion_file,
        skill=args.skill,
        key_body_ids=[],
        device=args.device,
    )
    motion_id = int(args.motion_id)
    if motion_id < 0 or motion_id >= motion_lib.num_motions():
        raise ValueError(f"motion_id {motion_id} out of range [0, {motion_lib.num_motions()})")

    motion = motion_lib._motions[motion_id]
    fps = float(motion["fps"])
    dt = 1.0 / float(args.fps or fps)
    motion_len = float(motion_lib.get_motion_length(torch.tensor([motion_id], device=args.device))[0].item())
    frames = int(args.frames)
    times = torch.arange(frames, device=args.device, dtype=torch.float32) * dt
    times = torch.clamp(times, max=motion_len)
    motion_ids = torch.full((frames,), motion_id, device=args.device, dtype=torch.long)

    body_pos, body_rot, _, _ = motion_lib.get_motion_state_max(motion_ids, times)
    body_names = motion.get("link_body_list", [])
    if not body_names:
        raise ValueError("Selected motion has no link_body_list/body names.")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        body_names=np.asarray(body_names),
        body_pos=body_pos.detach().cpu().numpy().astype(np.float32),
        body_quat_xyzw=body_rot.detach().cpu().numpy().astype(np.float32),
        dt=np.asarray(dt, dtype=np.float32),
        motion_file=np.asarray(args.motion_file),
        skill=np.asarray(args.skill),
        motion_id=np.asarray(motion_id, dtype=np.int64),
    )
    print(f"saved AMP demo {frames} frames from {args.skill}[{motion_id}] -> {output}")
    return 0


def view(args: argparse.Namespace) -> int:
    import viser

    demo = load_rollout(Path(args.demo).resolve())
    policy = load_rollout(Path(args.policy).resolve())
    urdf_path = Path(args.urdf).resolve()

    demo_edges = parse_urdf_edges(urdf_path, demo["body_names"])
    policy_edges = parse_urdf_edges(urdf_path, policy["body_names"])
    demo_pos = align_body_pos(demo["body_pos"], args.demo_offset)
    policy_pos = align_body_pos(policy["body_pos"], args.policy_offset)
    num_frames = min(demo_pos.shape[0], policy_pos.shape[0])

    server = viser.ViserServer(host=args.host, port=args.port)
    if not args.no_grid:
        server.scene.add_grid("/ground", width=args.ground_size, height=args.ground_size)

    demo_colors = np.broadcast_to(np.asarray(args.demo_color, dtype=np.uint8), (demo_edges.shape[0], 2, 3)).copy()
    policy_colors = np.broadcast_to(np.asarray(args.policy_color, dtype=np.uint8), (policy_edges.shape[0], 2, 3)).copy()
    demo_lines = server.scene.add_line_segments(
        "/amp_demo/skeleton",
        points=np.zeros((demo_edges.shape[0], 2, 3), dtype=np.float32),
        colors=demo_colors,
        line_width=args.line_width,
    )
    policy_lines = server.scene.add_line_segments(
        "/policy/skeleton",
        points=np.zeros((policy_edges.shape[0], 2, 3), dtype=np.float32),
        colors=policy_colors,
        line_width=args.line_width,
    )

    status = server.gui.add_text("frame", initial_value="")
    playing = server.gui.add_checkbox("play", initial_value=True)
    frame_slider = server.gui.add_slider("frame", min=0, max=num_frames - 1, step=1, initial_value=0)

    def set_frame(frame_index: int) -> None:
        frame_index = int(np.clip(frame_index, 0, num_frames - 1))
        demo_lines.points = demo_pos[frame_index, demo_edges]
        policy_lines.points = policy_pos[frame_index, policy_edges]
        status.value = f"{frame_index}/{num_frames - 1}  green=AMP demo  orange=policy"

    @frame_slider.on_update
    def _(_) -> None:
        set_frame(int(frame_slider.value))

    set_frame(0)
    print(f"Viser demo-vs-policy skeleton running on http://{args.host}:{args.port}")
    next_frame_time = time.time()
    while True:
        if playing.value:
            now = time.time()
            if now >= next_frame_time:
                frame_slider.value = int((int(frame_slider.value) + 1) % num_frames)
                set_frame(int(frame_slider.value))
                next_frame_time = now + max(float(args.play_dt), 1.0e-3)
        time.sleep(0.005)


def parse_color(text: str) -> Tuple[int, int, int]:
    parts = [int(part) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be R,G,B")
    return tuple(int(np.clip(part, 0, 255)) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-demo", help="Export MotionLib AMP demo states as rollout npz.")
    export.add_argument("--motion-file", default="tokenhsi/data/dataset_g1_all_urdf_canonical_filtered.yaml")
    export.add_argument("--skill", default="loco")
    export.add_argument("--motion-id", type=int, default=0)
    export.add_argument("--frames", type=int, default=300)
    export.add_argument("--fps", type=float, default=30.0)
    export.add_argument("--device", default="cpu")
    export.add_argument("--output", default="/tmp/g1_amp_demo_loco_motion0.npz")
    export.set_defaults(func=export_demo)

    view_cmd = subparsers.add_parser("view", help="View AMP demo and policy rollout skeletons together.")
    view_cmd.add_argument("--demo", default="/tmp/g1_amp_demo_loco_motion0.npz")
    view_cmd.add_argument("--policy", default="/tmp/g1_loco_reward_latest_1000_rollout.npz")
    view_cmd.add_argument("--urdf", default="tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf")
    view_cmd.add_argument("--host", default="127.0.0.1")
    view_cmd.add_argument("--port", type=int, default=8090)
    view_cmd.add_argument("--play-dt", type=float, default=1.0 / 30.0)
    view_cmd.add_argument("--ground-size", type=float, default=4.0)
    view_cmd.add_argument("--no-grid", action="store_true")
    view_cmd.add_argument("--demo-offset", type=float, nargs=3, default=[0.0, 0.8, 0.0])
    view_cmd.add_argument("--policy-offset", type=float, nargs=3, default=[0.0, -0.8, 0.0])
    view_cmd.add_argument("--demo-color", type=parse_color, default=(40, 190, 95))
    view_cmd.add_argument("--policy-color", type=parse_color, default=(235, 135, 35))
    view_cmd.add_argument("--line-width", type=float, default=3.0)
    view_cmd.set_defaults(func=view)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
