#!/usr/bin/env python3
"""Visualize the G1 BrainCo-hand URDF in viser with interactive joint sliders.

Usage:
    python tools/visualize_g1_urdf.py
    python tools/visualize_g1_urdf.py --urdf assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh
import viser


@dataclass(frozen=True)
class VisualSpec:
    mesh_path: Path
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    mimic_joint: Optional[str]
    mimic_multiplier: float
    mimic_offset: float


@dataclass(frozen=True)
class LinkSpec:
    name: str
    visuals: Tuple[VisualSpec, ...]


def parse_vec(text: Optional[str], default: Iterable[float]) -> np.ndarray:
    if not text:
        return np.array(list(default), dtype=np.float64)
    return np.array([float(part) for part in text.split()], dtype=np.float64)


def rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return rot_z(float(rpy[2])) @ rot_y(float(rpy[1])) @ rot_x(float(rpy[0]))


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def parse_urdf(urdf_path: Path) -> Tuple[Dict[str, LinkSpec], Dict[str, JointSpec], str]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: Dict[str, LinkSpec] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.attrib["name"]
        visuals: List[VisualSpec] = []
        for visual_elem in link_elem.findall("visual"):
            origin_elem = visual_elem.find("origin")
            origin_xyz = parse_vec(origin_elem.attrib.get("xyz") if origin_elem is not None else None, [0.0, 0.0, 0.0])
            origin_rpy = parse_vec(origin_elem.attrib.get("rpy") if origin_elem is not None else None, [0.0, 0.0, 0.0])
            geometry_elem = visual_elem.find("geometry")
            if geometry_elem is None:
                continue
            mesh_elem = geometry_elem.find("mesh")
            if mesh_elem is None:
                continue
            mesh_filename = mesh_elem.attrib.get("filename")
            if not mesh_filename:
                continue
            scale = parse_vec(mesh_elem.attrib.get("scale"), [1.0, 1.0, 1.0])
            visuals.append(
                VisualSpec(
                    mesh_path=(urdf_path.parent / mesh_filename).resolve(),
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    scale=scale,
                )
            )
        links[link_name] = LinkSpec(name=link_name, visuals=tuple(visuals))

    joints: Dict[str, JointSpec] = {}
    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "fixed")
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        limit_elem = joint_elem.find("limit")
        mimic_elem = joint_elem.find("mimic")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            raise RuntimeError(f"Malformed joint '{joint_name}': missing parent or child link")
        joints[joint_name] = JointSpec(
            name=joint_name,
            joint_type=joint_type,
            parent=parent_elem.attrib["link"],
            child=child_elem.attrib["link"],
            origin_xyz=parse_vec(origin_elem.attrib.get("xyz") if origin_elem is not None else None, [0.0, 0.0, 0.0]),
            origin_rpy=parse_vec(origin_elem.attrib.get("rpy") if origin_elem is not None else None, [0.0, 0.0, 0.0]),
            axis=parse_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, [0.0, 0.0, 1.0]),
            lower=float(limit_elem.attrib.get("lower", "0.0")) if limit_elem is not None else 0.0,
            upper=float(limit_elem.attrib.get("upper", "0.0")) if limit_elem is not None else 0.0,
            mimic_joint=mimic_elem.attrib.get("joint") if mimic_elem is not None else None,
            mimic_multiplier=float(mimic_elem.attrib.get("multiplier", "1.0")) if mimic_elem is not None else 1.0,
            mimic_offset=float(mimic_elem.attrib.get("offset", "0.0")) if mimic_elem is not None else 0.0,
        )

    child_links = {joint.child for joint in joints.values()}
    root_links = [link_name for link_name in links.keys() if link_name not in child_links]
    if not root_links:
        raise RuntimeError("Failed to identify a root link in the URDF")

    return links, joints, root_links[0]


def build_children_map(joints: Dict[str, JointSpec]) -> Dict[str, List[JointSpec]]:
    children: Dict[str, List[JointSpec]] = {}
    for joint in joints.values():
        children.setdefault(joint.parent, []).append(joint)
    for child_joints in children.values():
        child_joints.sort(key=lambda joint: joint.name)
    return children


def resolve_joint_values(joints: Dict[str, JointSpec], slider_values: Dict[str, float]) -> Dict[str, float]:
    resolved: Dict[str, float] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> float:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise RuntimeError(f"Cyclic mimic chain detected at joint '{name}'")
        visiting.add(name)
        joint = joints[name]
        if joint.mimic_joint:
            value = joint.mimic_multiplier * resolve(joint.mimic_joint) + joint.mimic_offset
        else:
            value = slider_values.get(name, 0.0)
        resolved[name] = value
        visiting.remove(name)
        return value

    for joint_name in joints:
        resolve(joint_name)
    return resolved


def compute_link_transforms(
    root_link: str,
    children_map: Dict[str, List[JointSpec]],
    joint_values: Dict[str, float],
) -> Dict[str, np.ndarray]:
    link_transforms: Dict[str, np.ndarray] = {root_link: np.eye(4, dtype=np.float64)}

    def visit(link_name: str) -> None:
        parent_transform = link_transforms[link_name]
        for joint in children_map.get(link_name, []):
            joint_transform = parent_transform @ make_transform(joint.origin_xyz, joint.origin_rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                motion_transform = np.eye(4, dtype=np.float64)
                motion_transform[:3, :3] = axis_angle_to_matrix(joint.axis, joint_values[joint.name])
                child_transform = joint_transform @ motion_transform
            elif joint.joint_type == "prismatic":
                motion_transform = np.eye(4, dtype=np.float64)
                axis_norm = float(np.linalg.norm(joint.axis))
                if axis_norm > 1e-12:
                    motion_transform[:3, 3] = joint.axis / axis_norm * joint_values[joint.name]
                child_transform = joint_transform @ motion_transform
            else:
                child_transform = joint_transform
            link_transforms[joint.child] = child_transform
            visit(joint.child)

    visit(root_link)
    return link_transforms


def mesh_cache_load(mesh_path: Path, scale: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not np.allclose(scale, np.ones(3)):
        mesh = mesh.copy()
        mesh.apply_scale(scale)
    return mesh


def quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
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
    return np.array([w, x, y, z], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        default="assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf",
        help="Path to the URDF to visualize.",
    )
    parser.add_argument("--ground-size", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)

    links, joints, root_link = parse_urdf(urdf_path)
    children_map = build_children_map(joints)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=args.ground_size, height=args.ground_size)

    mesh_handles: List[Tuple[str, int, object]] = []
    mesh_cache: Dict[Tuple[Path, Tuple[float, float, float]], trimesh.Trimesh] = {}

    for link_name, link_spec in links.items():
        for visual_index, visual in enumerate(link_spec.visuals):
            cache_key = (visual.mesh_path, tuple(float(value) for value in visual.scale))
            mesh = mesh_cache.get(cache_key)
            if mesh is None:
                mesh = mesh_cache_load(visual.mesh_path, visual.scale)
                mesh_cache[cache_key] = mesh
            handle = server.scene.add_mesh_trimesh(f"/links/{link_name}/visual_{visual_index}", mesh)
            mesh_handles.append((link_name, visual_index, handle))

    slider_handles = {}
    with server.gui.add_folder("Joints"):
        for joint in joints.values():
            if joint.joint_type not in {"revolute", "continuous", "prismatic"}:
                continue
            if joint.mimic_joint is not None:
                continue
            if math.isclose(joint.lower, joint.upper):
                continue
            if joint.lower <= 0.0 <= joint.upper:
                initial_value = 0.0
            else:
                initial_value = 0.5 * (joint.lower + joint.upper)
            slider_handles[joint.name] = server.gui.add_slider(
                label=joint.name,
                min=float(joint.lower),
                max=float(joint.upper),
                step=1e-3,
                initial_value=float(initial_value),
            )

    def sync_scene() -> None:
        slider_values = {name: slider.value for name, slider in slider_handles.items()}
        joint_values = resolve_joint_values(joints, slider_values)
        link_transforms = compute_link_transforms(root_link, children_map, joint_values)

        for link_name, visual_index, handle in mesh_handles:
            link_transform = link_transforms[link_name]
            visual = links[link_name].visuals[visual_index]
            visual_transform = link_transform @ make_transform(visual.origin_xyz, visual.origin_rpy)
            handle.position = visual_transform[:3, 3]
            handle.wxyz = quat_from_matrix(visual_transform[:3, :3])

    sync_scene()

    for slider in slider_handles.values():
        slider.on_update(lambda _event: sync_scene())

    with server.gui.add_folder("Controls"):
        reset_button = server.gui.add_button("Reset joints")

        @reset_button.on_click
        def _(_event) -> None:
            for slider in slider_handles.values():
                slider.value = 0.0
            sync_scene()

    print(f"Viser server running for {urdf_path}")
    print("Open the printed URL in your browser.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())