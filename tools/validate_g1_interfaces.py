#!/usr/bin/env python3
"""Offline interface validation for the G1 + Dex3 TokenHSI port.

This script intentionally avoids importing IsaacGym. It validates the contracts
that can be checked from files and pure PyTorch:

- env/train yaml consistency
- MJCF mesh, body, joint and action dimensions
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
EXPECTED_ASSET = "mjcf/g1_mocap_29dof_with_hands.xml"
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
    }


def load_motion_file(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def validate_motion_schema(data: Dict[str, Any], path: Path, expected_dof: int, expected_bodies: List[str]) -> None:
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
    if list(data["link_body_list"]) != expected_bodies:
        fail("{} link_body_list does not match MJCF body list".format(path))
    if data["local_body_pos"].shape[1] != len(expected_bodies):
        fail("{} local_body_pos body count mismatch".format(path))


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

    xml_path = ROOT / "tokenhsi/data/assets" / env["asset"]["assetFileName"]
    mjcf = parse_mjcf(xml_path)
    if mjcf["missing_meshes"]:
        fail("missing meshes: {}".format(mjcf["missing_meshes"][:10]))
    if len(mjcf["joint_names"]) != 43:
        fail("expected 43 actuated joints, got {}".format(len(mjcf["joint_names"])))
    if len(mjcf["body_names"]) != 52:
        fail("expected 52 bodies, got {}".format(len(mjcf["body_names"])))

    key_body_ids = []
    for name in env["keyBodies"]:
        if name not in mjcf["body_names"]:
            fail("key body {} missing in MJCF".format(name))
        key_body_ids.append(mjcf["body_names"].index(name))
    for name in env["contactBodies"]:
        if name not in mjcf["body_names"]:
            fail("contact body {} missing in MJCF".format(name))

    missing_entries = []
    motion_count = 0
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
            validate_motion_schema(load_motion_file(motion_path), motion_path, 43, mjcf["body_names"])
    if missing_entries:
        fail("missing motion entries: {}".format(missing_entries[:10]))

    self_obs_size = 1 + 52 * (3 + 6 + 3 + 3) - 3
    task_obs_size = 20 + 38 + 42 + 27 + 4
    obs_size = self_obs_size + task_obs_size
    amp_step_size = 13 + 43 * 6 + 43 + 3 * len(env["keyBodies"]) + 4
    amp_obs_size = amp_step_size * env["numAMPObsSteps"]

    if (self_obs_size, task_obs_size, obs_size, amp_step_size, amp_obs_size) != (778, 131, 909, 330, 3300):
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
    print("dimensions: self_obs=778 task_obs=131 obs=909 actions=43 amp_step=330 amp_obs=3300")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
