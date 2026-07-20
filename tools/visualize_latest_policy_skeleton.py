#!/usr/bin/env python3
"""Record the latest TokenHSI-G1 checkpoint rollout and view it as a viser skeleton."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_CFG_TRAIN = "tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml"
DEFAULT_CFG_ENV = "tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml"
DEFAULT_MOTION_FILE = "tokenhsi/data/dataset_g1_all_urdf_canonical_filtered.yaml"
DEFAULT_URDF = "tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf"


def find_latest_checkpoint(output_root: Path) -> Path:
    checkpoints = [
        path
        for path in output_root.rglob("*.pth")
        if path.is_file() and not path.name.endswith(".part")
    ]
    if not checkpoints:
        raise FileNotFoundError("No .pth checkpoint found under {}".format(output_root))
    return max(checkpoints, key=lambda path: (path.stat().st_mtime, str(path)))


def load_rollout(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        body_names = tuple(str(name) for name in data["body_names"].tolist())
        body_pos = np.asarray(data["body_pos"], dtype=np.float32)
        body_quat_xyzw = np.asarray(data["body_quat_xyzw"], dtype=np.float32)
        dt = float(np.asarray(data["dt"]).reshape(()))
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError("body_pos must be [frames, bodies, 3], got {}".format(body_pos.shape))
    if body_quat_xyzw.ndim != 3 or body_quat_xyzw.shape[-1] != 4:
        raise ValueError("body_quat_xyzw must be [frames, bodies, 4], got {}".format(body_quat_xyzw.shape))
    if body_pos.shape[:2] != body_quat_xyzw.shape[:2]:
        raise ValueError("body_pos/body_quat frame or body count mismatch")
    if body_pos.shape[1] != len(body_names):
        raise ValueError("body_names has {} entries but rollout has {} bodies".format(len(body_names), body_pos.shape[1]))
    return {
        "body_names": body_names,
        "body_pos": body_pos,
        "body_quat_xyzw": body_quat_xyzw,
        "dt": dt,
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
        raise ValueError("No URDF skeleton edges matched rollout body names from {}".format(urdf_path))
    return np.asarray(edges, dtype=np.int64)


def align_body_pos(body_pos: np.ndarray, offset: Sequence[float]) -> np.ndarray:
    aligned = np.array(body_pos, dtype=np.float32, copy=True)
    root_xy = aligned[0, 0, 0:2].copy()
    aligned[..., 0:2] -= root_xy
    aligned += np.asarray(offset, dtype=np.float32).reshape(1, 1, 3)
    return aligned


def build_record_command(args: argparse.Namespace, checkpoint: Path, rollout: Path) -> Tuple[List[str], Dict[str, str]]:
    cmd = [
        args.python,
        "./tokenhsi/run.py",
        "--task",
        args.task,
        "--cfg_train",
        args.cfg_train,
        "--cfg_env",
        args.cfg_env,
        "--motion_file",
        args.motion_file,
        "--num_envs",
        str(args.num_envs),
        "--rl_device",
        args.rl_device,
        "--sim_device",
        args.sim_device,
        "--pipeline",
        args.pipeline,
        "--graphics_device_id",
        str(args.graphics_device_id),
        "--checkpoint",
        str(checkpoint),
        "--output_path",
        args.output_path,
        "--play",
        "--headless",
    ]
    env = os.environ.copy()
    env["TOKENHSI_RECORD_ROLLOUT_NPZ"] = str(rollout)
    env["TOKENHSI_RECORD_ROLLOUT_ENV_ID"] = str(args.env_id)
    env["TOKENHSI_RECORD_ROLLOUT_MAX_FRAMES"] = str(args.steps)
    return cmd, env


def record(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    checkpoint = find_latest_checkpoint(output_root) if args.checkpoint == "latest" else Path(args.checkpoint).resolve()
    rollout = Path(args.rollout).resolve()
    rollout.parent.mkdir(parents=True, exist_ok=True)
    cmd, env = build_record_command(args, checkpoint, rollout)
    print("checkpoint: {}".format(checkpoint))
    print("rollout: {}".format(rollout))
    print("command: {}".format(" ".join(cmd)))
    subprocess.run(cmd, cwd=Path(args.repo).resolve(), env=env, check=True)
    if not rollout.exists():
        raise FileNotFoundError("Rollout recorder did not create {}".format(rollout))
    return 0


def view(args: argparse.Namespace) -> int:
    import viser

    rollout = load_rollout(Path(args.rollout).resolve())
    edges = parse_urdf_edges(Path(args.urdf).resolve(), rollout["body_names"])
    body_pos = align_body_pos(rollout["body_pos"], args.offset)
    colors = np.broadcast_to(np.asarray(args.color, dtype=np.uint8), (edges.shape[0], 2, 3)).copy()

    server = viser.ViserServer(host=args.host, port=args.port)
    if args.grid:
        server.scene.add_grid("/ground", width=args.ground_size, height=args.ground_size)
    lines = server.scene.add_line_segments(
        "/policy/skeleton",
        points=np.zeros((edges.shape[0], 2, 3), dtype=np.float32),
        colors=colors,
        line_width=args.line_width,
    )
    status = server.gui.add_text("frame", initial_value="")
    playing = server.gui.add_checkbox("play", initial_value=True)
    frame_slider = server.gui.add_slider("frame", min=0, max=body_pos.shape[0] - 1, step=1, initial_value=0)

    def set_frame(frame_index: int) -> None:
        frame_index = int(np.clip(frame_index, 0, body_pos.shape[0] - 1))
        lines.points = body_pos[frame_index, edges]
        root_z = float(body_pos[frame_index, 0, 2])
        status.value = "{}/{}  root_z={:.4f}".format(frame_index, body_pos.shape[0] - 1, root_z)

    @frame_slider.on_update
    def _(_) -> None:
        set_frame(int(frame_slider.value))

    set_frame(0)
    print("Viser policy skeleton running on http://{}:{}".format(args.host, args.port))
    next_frame_time = time.time()
    while True:
        if playing.value:
            now = time.time()
            if now >= next_frame_time:
                frame_slider.value = int((int(frame_slider.value) + 1) % body_pos.shape[0])
                set_frame(int(frame_slider.value))
                next_frame_time = now + max(float(args.play_dt), 1.0e-3)
        time.sleep(0.005)


def record_and_view(args: argparse.Namespace) -> int:
    record(args)
    return view(args)


def parse_color(text: str) -> Tuple[int, int, int]:
    parts = [int(part) for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be R,G,B")
    return tuple(int(np.clip(part, 0, 255)) for part in parts)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--rollout", default="/tmp/g1_latest_policy_rollout.npz")
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", "python"))
    parser.add_argument("--task", default="HumanoidTrajSitCarryClimb")
    parser.add_argument("--cfg-train", default=DEFAULT_CFG_TRAIN)
    parser.add_argument("--cfg-env", default=DEFAULT_CFG_ENV)
    parser.add_argument("--motion-file", default=DEFAULT_MOTION_FILE)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--output-path", default="output/latest_policy_rollout_eval")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--pipeline", default="gpu")
    parser.add_argument("--graphics-device-id", type=int, default=0)


def add_view_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--play-dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--color", type=parse_color, default=(235, 135, 35))
    parser.add_argument("--line-width", type=float, default=3.0)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--ground-size", type=float, default=4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="Record one policy rollout from the latest checkpoint.")
    add_common_args(record_parser)
    record_parser.set_defaults(func=record)

    view_parser = subparsers.add_parser("view", help="View a saved policy rollout as a skeleton.")
    view_parser.add_argument("--rollout", default="/tmp/g1_latest_policy_rollout.npz")
    add_view_args(view_parser)
    view_parser.set_defaults(func=view)

    run_parser = subparsers.add_parser("run", help="Record from the latest checkpoint, then open the skeleton viewer.")
    add_common_args(run_parser)
    add_view_args(run_parser)
    run_parser.set_defaults(func=record_and_view)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
