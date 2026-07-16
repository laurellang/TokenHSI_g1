#!/usr/bin/env python3
"""Visualize the G1 BrainCo-hand URDF in viser with interactive joint sliders.

Example:
    conda activate tokenhsi_g1
    python tools/visualize_g1_brainco_urdf.py

Install runtime viewer deps if needed:
    pip install viser
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = Path("assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf")


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Transform:
    matrix: np.ndarray

    @staticmethod
    def identity() -> "Transform":
        return Transform(np.eye(4))

    @staticmethod
    def from_xyz_rpy(xyz: Vector3, rpy: Vector3) -> "Transform":
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

        matrix = np.eye(4)
        matrix[:3, :3] = rz @ ry @ rx
        matrix[:3, 3] = np.asarray(xyz, dtype=float)
        return Transform(matrix)

    def __matmul__(self, other: "Transform") -> "Transform":
        return Transform(self.matrix @ other.matrix)


@dataclass(frozen=True)
class Visual:
    link: str
    name: str
    mesh_path: Path
    origin: Transform
    scale: Vector3 = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: Transform
    axis: Vector3
    limit_lower: Optional[float]
    limit_upper: Optional[float]


@dataclass
class UrdfModel:
    urdf_path: Path
    root_link: str
    links: List[str]
    joints: List[Joint]
    visuals_by_link: Dict[str, List[Visual]] = field(default_factory=dict)

    @property
    def actuated_joints(self) -> List[Joint]:
        return [joint for joint in self.joints if joint.joint_type in {"revolute", "continuous", "prismatic"}]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="URDF path to visualize.")
    parser.add_argument("--host", default="127.0.0.1", help="viser server host.")
    parser.add_argument("--port", type=int, default=8080, help="viser server port.")
    parser.add_argument("--no-grid", action="store_true", help="Do not show the ground grid.")
    return parser.parse_args(argv)


def resolve_urdf_path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _float_tuple(text: Optional[str], default: Vector3) -> Vector3:
    if text is None:
        return default
    values = tuple(float(part) for part in text.split())
    if len(values) != 3:
        raise ValueError(f"Expected 3 floats, got {text!r}")
    return values  # type: ignore[return-value]


def _origin_from_xml(element: ET.Element) -> Transform:
    origin = element.find("origin")
    if origin is None:
        return Transform.identity()
    xyz = _float_tuple(origin.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _float_tuple(origin.get("rpy"), (0.0, 0.0, 0.0))
    return Transform.from_xyz_rpy(xyz, rpy)


def _mesh_path(filename: str, urdf_dir: Path) -> Path:
    if filename.startswith("package://"):
        filename = filename.removeprefix("package://")
        parts = Path(filename).parts
        filename = str(Path(*parts[1:])) if len(parts) > 1 else parts[0]
    path = Path(filename)
    if not path.is_absolute():
        path = urdf_dir / path
    return path.resolve()


def _parse_visuals(link: ET.Element, link_name: str, urdf_dir: Path) -> List[Visual]:
    visuals = []
    for idx, visual_xml in enumerate(link.findall("visual")):
        mesh_xml = visual_xml.find("geometry/mesh")
        if mesh_xml is None or mesh_xml.get("filename") is None:
            continue
        name = visual_xml.get("name") or f"{link_name}_visual_{idx}"
        visuals.append(
            Visual(
                link=link_name,
                name=name,
                mesh_path=_mesh_path(mesh_xml.get("filename", ""), urdf_dir),
                origin=_origin_from_xml(visual_xml),
                scale=_float_tuple(mesh_xml.get("scale"), (1.0, 1.0, 1.0)),
            )
        )
    return visuals


def _joint_axis(joint_xml: ET.Element) -> Vector3:
    axis = joint_xml.find("axis")
    return _float_tuple(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0))


def load_urdf_model(path: Path) -> UrdfModel:
    urdf_path = resolve_urdf_path(path)
    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)

    root_xml = ET.parse(urdf_path).getroot()
    links = [link.get("name", "") for link in root_xml.findall("link") if link.get("name")]
    visuals_by_link = {
        link_name: visuals
        for link_xml in root_xml.findall("link")
        if (link_name := link_xml.get("name"))
        for visuals in [_parse_visuals(link_xml, link_name, urdf_path.parent)]
    }

    joints = []
    child_links = set()
    for joint_xml in root_xml.findall("joint"):
        parent = joint_xml.find("parent")
        child = joint_xml.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link", "")
        child_name = child.get("link", "")
        child_links.add(child_name)

        limit = joint_xml.find("limit")
        lower = float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None
        upper = float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None

        joints.append(
            Joint(
                name=joint_xml.get("name", ""),
                joint_type=joint_xml.get("type", "fixed"),
                parent=parent_name,
                child=child_name,
                origin=_origin_from_xml(joint_xml),
                axis=_joint_axis(joint_xml),
                limit_lower=lower,
                limit_upper=upper,
            )
        )

    roots = [link for link in links if link not in child_links]
    if not roots:
        raise ValueError(f"No root link found in {urdf_path}")
    return UrdfModel(urdf_path=urdf_path, root_link=roots[0], links=links, joints=joints, visuals_by_link=visuals_by_link)


def slider_range(joint: Joint) -> Tuple[float, float]:
    if joint.limit_lower is not None and joint.limit_upper is not None and joint.limit_lower < joint.limit_upper:
        return joint.limit_lower, joint.limit_upper
    if joint.joint_type == "prismatic":
        return -0.25, 0.25
    return -math.pi, math.pi


def _axis_angle(axis: Vector3, angle: float) -> Transform:
    axis_np = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_np)
    if norm == 0.0:
        return Transform.identity()
    x, y, z = axis_np / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return Transform(matrix)


def _translation(axis: Vector3, distance: float) -> Transform:
    axis_np = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_np)
    matrix = np.eye(4)
    if norm != 0.0:
        matrix[:3, 3] = axis_np / norm * distance
    return Transform(matrix)


def compute_link_transforms(model: UrdfModel, joint_values: Dict[str, float]) -> Dict[str, Transform]:
    children_by_parent: Dict[str, List[Joint]] = {}
    for joint in model.joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)

    transforms = {model.root_link: Transform.identity()}

    def visit(parent: str) -> None:
        parent_transform = transforms[parent]
        for joint in children_by_parent.get(parent, []):
            value = joint_values.get(joint.name, 0.0)
            motion = Transform.identity()
            if joint.joint_type in {"revolute", "continuous"}:
                motion = _axis_angle(joint.axis, value)
            elif joint.joint_type == "prismatic":
                motion = _translation(joint.axis, value)
            transforms[joint.child] = parent_transform @ joint.origin @ motion
            visit(joint.child)

    visit(model.root_link)
    return transforms


def _load_trimesh(path: Path, scale: Vector3):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    mesh = mesh.copy()
    mesh.vertices *= np.asarray(scale, dtype=float)
    return mesh


def run_viewer(args: argparse.Namespace) -> None:
    try:
        import viser
        import viser.transforms as vtf
    except ModuleNotFoundError as exc:
        missing = exc.name or "viser"
        raise SystemExit(f"Missing dependency {missing!r}. Install it with: pip install viser") from exc

    model = load_urdf_model(args.urdf)
    server = viser.ViserServer(host=args.host, port=args.port)
    if not args.no_grid:
        server.scene.add_grid("/ground", width=4, height=4)

    joint_values = {joint.name: 0.0 for joint in model.actuated_joints}
    link_transforms = compute_link_transforms(model, joint_values)
    mesh_handles = []

    for link_name, visuals in model.visuals_by_link.items():
        for visual_id, visual in enumerate(visuals):
            if not visual.mesh_path.exists():
                print(f"[warn] missing mesh: {visual.mesh_path}")
                continue
            mesh = _load_trimesh(visual.mesh_path, visual.scale)
            handle = server.scene.add_mesh_trimesh(f"/robot/{link_name}/{visual_id}_{visual.name}", mesh)
            mesh_handles.append((link_name, visual.origin, handle))

    def sync_visuals() -> None:
        transforms = compute_link_transforms(model, joint_values)
        for link_name, visual_origin, handle in mesh_handles:
            world_from_visual = transforms.get(link_name, Transform.identity()) @ visual_origin
            handle.position = world_from_visual.matrix[:3, 3]
            handle.wxyz = vtf.SO3.from_matrix(world_from_visual.matrix[:3, :3]).wxyz

    with server.gui.add_folder("Joints"):
        for joint in model.actuated_joints:
            lower, upper = slider_range(joint)
            slider = server.gui.add_slider(
                label=joint.name,
                min=float(lower),
                max=float(upper),
                step=1e-3,
                initial_value=0.0,
            )

            @slider.on_update
            def _(_, joint_name: str = joint.name, slider_handle=slider) -> None:
                joint_values[joint_name] = float(slider_handle.value)
                sync_visuals()

    with server.gui.add_folder("Controls"):
        reset_button = server.gui.add_button("Reset joints to 0")

        @reset_button.on_click
        def _(_) -> None:
            for name in joint_values:
                joint_values[name] = 0.0
            sync_visuals()

    sync_visuals()
    print(f"Loaded {model.urdf_path}")
    print(f"Root link: {model.root_link}; joints: {len(model.joints)}; visuals: {len(mesh_handles)}")
    print("Viser server running. Open the printed URL in your browser.")
    while True:
        time.sleep(1.0)


def main() -> int:
    run_viewer(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
