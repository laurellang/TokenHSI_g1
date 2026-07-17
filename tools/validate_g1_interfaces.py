#!/usr/bin/env python3
"""Offline interface validation for the G1 + Dex3 TokenHSI port.

This script intentionally avoids importing IsaacGym. It validates the contracts
that can be checked from files and pure PyTorch:

- env/train yaml consistency
- robot asset mesh, body, joint and action dimensions
- GMR retargeted motion yaml and pkl schema
- GMRRobotMotionLib sampling/state shapes on CPU
- transformer policy/discriminator tensor interfaces on CPU
"""

from __future__ import annotations

import argparse
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml"
TRAIN_CFG = ROOT / "tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml"
DATA_YAML = ROOT / "tokenhsi/data/dataset_g1_all.yaml"
EXPECTED_ASSET = "mjcf/g1_dex3_ori.urdf"
RETARGET_REFERENCE_ASSET = "mjcf/g1_mocap_29dof_with_hands.xml"
EXPECTED_MOTION = "tokenhsi/data/dataset_g1_all.yaml"
EXPECTED_TASKS = ["traj", "sit", "carry", "climb"]
EXPECTED_SKILLS = [
    "loco",
    "loco_sit",
    "sit",
    "loco_carry",
    "pickUp",
    "carryWith",
    "putDown",
    "omomo",
    "loco_climb",
    "climb",
    "climbNoRSI",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_mjcf(xml_path: Path) -> Dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        fail("{} has no <compiler>".format(xml_path))
    meshdir = compiler.attrib.get("meshdir")
    if not meshdir:
        fail("{} compiler has no meshdir".format(xml_path))

    mesh_root = (xml_path.parent / meshdir).resolve()
    meshes = [m.attrib["file"] for m in root.findall(".//mesh") if "file" in m.attrib]
    missing_meshes = [m for m in meshes if not (mesh_root / m).exists()]

    body_names: List[str] = []
    joint_names: List[str] = []
    dof_body_ids: List[int] = []
    joint_ranges: List[List[float]] = []

    worldbody = root.find("worldbody")
    if worldbody is None:
        fail("{} has no <worldbody>".format(xml_path))

    def walk(body: ET.Element) -> None:
        body_id = len(body_names)
        body_names.append(body.attrib.get("name", ""))
        for joint in body.findall("joint"):
            name = joint.attrib.get("name")
            if name is not None and joint.attrib.get("type") != "free":
                joint_names.append(name)
                dof_body_ids.append(body_id)
                joint_range = joint.attrib.get("range")
                if joint_range is None:
                    joint_ranges.append([float("nan"), float("nan")])
                else:
                    lower, upper = joint_range.split()
                    joint_ranges.append([float(lower), float(upper)])
        for child in body.findall("body"):
            walk(child)

    for body in worldbody.findall("body"):
        walk(body)

    return {
        "mesh_root": mesh_root,
        "mesh_count": len(meshes),
        "missing_meshes": missing_meshes,
        "body_names": body_names,
        "joint_names": joint_names,
        "dof_body_ids": dof_body_ids,
        "joint_ranges": joint_ranges,
    }


def parse_urdf(urdf_path: Path) -> Dict[str, Any]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if root.tag != "robot":
        fail("{} is not a URDF robot asset".format(urdf_path))

    meshes = [m.attrib["filename"] for m in root.findall(".//mesh") if "filename" in m.attrib]
    missing_meshes = [m for m in meshes if not (urdf_path.parent / m).exists()]

    body_names = [link.attrib["name"] for link in root.findall("link")]
    body_index = {name: idx for idx, name in enumerate(body_names)}
    joint_names: List[str] = []
    dof_body_ids: List[int] = []
    joint_ranges: List[List[float]] = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        child = joint.find("child")
        if child is None:
            fail("{} joint {} has no child link".format(urdf_path, joint.attrib.get("name", "")))
        child_name = child.attrib["link"]
        if child_name not in body_index:
            fail("{} joint {} child link {} missing".format(urdf_path, joint.attrib.get("name", ""), child_name))
        limit = joint.find("limit")
        if limit is None:
            fail("{} joint {} has no limit".format(urdf_path, joint.attrib["name"]))
        joint_names.append(joint.attrib["name"])
        dof_body_ids.append(body_index[child_name])
        joint_ranges.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])

    return {
        "mesh_root": urdf_path.parent,
        "mesh_count": len(meshes),
        "missing_meshes": missing_meshes,
        "body_names": body_names,
        "joint_names": joint_names,
        "dof_body_ids": dof_body_ids,
        "joint_ranges": joint_ranges,
    }


def parse_robot_asset(asset_path: Path) -> Dict[str, Any]:
    if asset_path.suffix == ".urdf":
        return parse_urdf(asset_path)
    if asset_path.suffix == ".xml":
        return parse_mjcf(asset_path)
    fail("unsupported robot asset extension: {}".format(asset_path))


def load_motion_file(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def validate_motion_schema(data: Dict[str, Any], path: Path, expected_dof: int) -> List[str]:
    for key in ["fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"]:
        if key not in data:
            fail("{} missing key {}".format(path, key))

    frame_count = data["root_pos"].shape[0]
    if data["root_rot"].shape[0] != frame_count:
        fail("{} root_rot frame count mismatch".format(path))
    if data["dof_pos"].shape[0] != frame_count:
        fail("{} dof_pos frame count mismatch".format(path))
    if data["local_body_pos"].shape[0] != frame_count:
        fail("{} local_body_pos frame count mismatch".format(path))
    if data["dof_pos"].shape[-1] != expected_dof:
        fail("{} has dof {}, expected {}".format(path, data["dof_pos"].shape[-1], expected_dof))
    motion_bodies = list(data["link_body_list"])
    if data["local_body_pos"].shape[1] != len(motion_bodies):
        fail("{} local_body_pos body count mismatch".format(path))
    return motion_bodies


def validate_body_schema_mapping(asset_bodies: List[str], motion_bodies: List[str], key_bodies: List[str], contact_bodies: List[str]) -> None:
    motion_body_set = set(motion_bodies)
    asset_body_set = set(asset_bodies)
    for name in key_bodies:
        if name not in asset_body_set:
            fail("key body {} missing in robot asset".format(name))
        if name not in motion_body_set:
            fail("key body {} missing in motion schema".format(name))
    for name in contact_bodies:
        if name not in asset_body_set:
            fail("contact body {} missing in robot asset".format(name))

    common_bodies = asset_body_set.intersection(motion_body_set)
    if len(common_bodies) < len(key_bodies) + 40:
        fail("asset/motion body schemas have too little overlap: {}".format(len(common_bodies)))


def validate_dof_alignment(robot_asset: Dict[str, Any]) -> None:
    reference_path = ROOT / "tokenhsi/data/assets" / RETARGET_REFERENCE_ASSET
    reference_asset = parse_robot_asset(reference_path)

    if robot_asset["joint_names"] != reference_asset["joint_names"]:
        mismatches = [
            (idx, actual, expected)
            for idx, (actual, expected) in enumerate(zip(robot_asset["joint_names"], reference_asset["joint_names"]))
            if actual != expected
        ]
        fail("DOF order mismatch with {}: {}".format(RETARGET_REFERENCE_ASSET, mismatches[:10]))

    range_mismatches = []
    for idx, (actual_range, expected_range) in enumerate(zip(robot_asset["joint_ranges"], reference_asset["joint_ranges"])):
        if abs(actual_range[0] - expected_range[0]) > 1e-5 or abs(actual_range[1] - expected_range[1]) > 1e-5:
            range_mismatches.append(
                (idx, robot_asset["joint_names"][idx], actual_range, expected_range)
            )
    if range_mismatches:
        fail("DOF range mismatch with {}: {}".format(RETARGET_REFERENCE_ASSET, range_mismatches[:10]))

    expected_left_hand = [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ]
    expected_right_hand = [
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ]
    joint_names = robot_asset["joint_names"]
    if joint_names[22:29] != expected_left_hand:
        fail("left DEX3 hand DOF slice mismatch: {}".format(joint_names[22:29]))
    if joint_names[36:43] != expected_right_hand:
        fail("right DEX3 hand DOF slice mismatch: {}".format(joint_names[36:43]))


def validate_object_assets(env: Dict[str, Any]) -> None:
    expected = {
        "sit": env["sit"]["objCategories"],
        "climb": env["climb"]["objCategories"],
    }
    data_root = ROOT / "tokenhsi/data"
    for task_name, categories in expected.items():
        dataset_root = data_root / env[task_name]["objDatasetDir"]
        if not dataset_root.exists():
            fail("{} object root missing: {}".format(task_name, dataset_root))
        for mode in ["train", "test"]:
            for category in categories:
                category_dir = dataset_root / mode / category
                if not category_dir.exists():
                    fail("{} object category missing: {}".format(task_name, category_dir))
                object_dirs = [p for p in category_dir.iterdir() if p.is_dir()]
                if not object_dirs:
                    fail("{} object category has no objects: {}".format(task_name, category_dir))
                for obj_dir in object_dirs:
                    for required in ["asset.urdf", "config.json"]:
                        if not (obj_dir / required).exists():
                            fail("{} missing {}".format(obj_dir, required))


def build_multi_task_info(device: str = "cpu") -> Dict[str, Any]:
    task_sizes = [20, 38, 42, 27]
    index = torch.cumsum(torch.tensor([0] + task_sizes, device=device), dim=0)
    mask = torch.zeros((len(task_sizes), sum(task_sizes)), dtype=torch.bool, device=device)
    for i in range(len(task_sizes)):
        mask[i, index[i] : index[i + 1]] = True
    return {
        "onehot_size": len(task_sizes),
        "tota_subtask_obs_size": sum(task_sizes),
        "each_subtask_obs_size": task_sizes,
        "each_subtask_obs_indx": index,
        "each_subtask_obs_mask": mask,
        "enable_task_mask_obs": True,
        "each_subtask_name": EXPECTED_TASKS,
    }


def validate_gmr_motion_lib(motion_yaml: str, key_body_ids: List[int]) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tokenhsi"))
    from tokenhsi.utils.gmr_robot_motion_lib import GMRRobotMotionLib

    lib = GMRRobotMotionLib(motion_file=motion_yaml, skill="loco", key_body_ids=key_body_ids, device="cpu")
    motion_ids = lib.sample_motions(4)
    motion_times = lib.sample_time(motion_ids)
    state = lib.get_motion_state(motion_ids, motion_times)
    shapes = [tuple(x.shape) for x in state]
    expected = [(4, 3), (4, 4), (4, 43), (4, 3), (4, 3), (4, 43), (4, len(key_body_ids), 3)]
    if shapes != expected:
        fail("GMRRobotMotionLib state shapes {} != {}".format(shapes, expected))

    max_state = lib.get_motion_state_max(motion_ids, motion_times)
    max_shapes = [tuple(x.shape) for x in max_state]
    expected_max = [(4, 52, 3), (4, 52, 4), (4, 52, 3), (4, 52, 3)]
    if max_shapes != expected_max:
        fail("GMRRobotMotionLib max-state shapes {} != {}".format(max_shapes, expected_max))


def validate_network_forward(train_cfg: Dict[str, Any], obs_size: int, self_obs_size: int, amp_obs_size: int) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tokenhsi"))
    from tokenhsi.learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder

    multi_task_info = build_multi_task_info()
    builder = AMPTransformerMultiTaskBuilder()
    builder.load(train_cfg["params"]["network"])
    net = builder.build(
        "amp",
        input_shape=(obs_size,),
        actions_num=43,
        amp_input_shape=(amp_obs_size,),
        self_obs_size=self_obs_size,
        task_obs_size=multi_task_info["tota_subtask_obs_size"] + multi_task_info["onehot_size"],
        multi_task_info=multi_task_info,
        device="cpu",
    )

    obs = torch.zeros((2, obs_size), dtype=torch.float32)
    onehot_start = self_obs_size + multi_task_info["tota_subtask_obs_size"]
    obs[:, onehot_start] = 1.0
    mu, sigma, value, _ = net({"obs": obs, "not_normalized_obs": obs})
    if tuple(mu.shape) != (2, 43):
        fail("network mu shape mismatch: {}".format(tuple(mu.shape)))
    if tuple(sigma.shape) != (43,):
        fail("network sigma shape mismatch: {}".format(tuple(sigma.shape)))
    if tuple(value.shape) != (2, 1):
        fail("network value shape mismatch: {}".format(tuple(value.shape)))
    disc = net.eval_disc(torch.zeros((2, amp_obs_size), dtype=torch.float32))
    if tuple(disc.shape) != (2, 1):
        fail("network disc shape mismatch: {}".format(tuple(disc.shape)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-network", action="store_true", help="Skip CPU transformer forward validation.")
    args = parser.parse_args()

    env_cfg = load_yaml(ENV_CFG)
    train_cfg = load_yaml(TRAIN_CFG)
    motion_cfg = load_yaml(DATA_YAML)
    env = env_cfg["env"]

    if env["asset"]["assetFileName"] != EXPECTED_ASSET:
        fail("cfg asset is {}, expected {}".format(env["asset"]["assetFileName"], EXPECTED_ASSET))
    if env["motion_file"] != EXPECTED_MOTION:
        fail("cfg motion_file is {}, expected {}".format(env["motion_file"], EXPECTED_MOTION))
    if env["task"] != EXPECTED_TASKS:
        fail("task list mismatch: {}".format(env["task"]))
    if env["skill"] != EXPECTED_SKILLS:
        fail("skill list mismatch: {}".format(env["skill"]))
    if motion_cfg.get("format") != "gmr_robot_motion":
        fail("motion yaml must use format: gmr_robot_motion")
    validate_object_assets(env)

    asset_path = ROOT / "tokenhsi/data/assets" / env["asset"]["assetFileName"]
    robot_asset = parse_robot_asset(asset_path)
    if robot_asset["missing_meshes"]:
        fail("missing meshes: {}".format(robot_asset["missing_meshes"][:10]))
    if len(robot_asset["joint_names"]) != 43:
        fail("expected 43 actuated joints, got {}".format(len(robot_asset["joint_names"])))
    if len(robot_asset["body_names"]) != 53:
        fail("expected 53 native URDF bodies, got {}".format(len(robot_asset["body_names"])))
    validate_dof_alignment(robot_asset)

    missing_entries = []
    motion_count = 0
    first_motion_bodies = None
    for skill in EXPECTED_SKILLS:
        entries = motion_cfg["motions"].get(skill)
        if not entries:
            fail("skill {} has no motion entries".format(skill))
        for entry in entries:
            motion_path = DATA_YAML.parent / entry["file"]
            motion_count += 1
            if not motion_path.exists():
                missing_entries.append(str(motion_path))
                continue
            motion_bodies = validate_motion_schema(load_motion_file(motion_path), motion_path, 43)
            if first_motion_bodies is None:
                first_motion_bodies = motion_bodies
            elif motion_bodies != first_motion_bodies:
                fail("{} motion body schema differs from first motion".format(motion_path))
    if missing_entries:
        fail("missing motion entries: {}".format(missing_entries[:10]))

    validate_body_schema_mapping(robot_asset["body_names"], first_motion_bodies, env["keyBodies"], env["contactBodies"])
    key_body_ids = [first_motion_bodies.index(name) for name in env["keyBodies"]]

    self_obs_size = 1 + len(robot_asset["body_names"]) * (3 + 6 + 3 + 3) - 3
    task_obs_size = 20 + 38 + 42 + 27 + 4
    obs_size = self_obs_size + task_obs_size
    amp_step_size = 13 + 43 * 6 + 43 + 3 * len(env["keyBodies"]) + 4
    amp_obs_size = amp_step_size * env["numAMPObsSteps"]

    if (self_obs_size, task_obs_size, obs_size, amp_step_size, amp_obs_size) != (793, 131, 924, 330, 3300):
        fail(
            "dimension mismatch: self={}, task={}, obs={}, amp_step={}, amp={}".format(
                self_obs_size, task_obs_size, obs_size, amp_step_size, amp_obs_size
            )
        )

    validate_gmr_motion_lib(env["motion_file"], key_body_ids)
    if not args.skip_network:
        validate_network_forward(train_cfg, obs_size, self_obs_size, amp_obs_size)

    print("G1 interface validation passed")
    print("asset:", env["asset"]["assetFileName"])
    print("motion yaml:", env["motion_file"])
    print("motion entry references checked:", motion_count)
    print("key body ids:", key_body_ids)
    print("dimensions: self_obs={} task_obs={} obs={} actions=43 amp_step={} amp_obs={}".format(
        self_obs_size,
        task_obs_size,
        obs_size,
        amp_step_size,
        amp_obs_size,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
