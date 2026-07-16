#!/usr/bin/env python3
"""Create a body-only G1 loco dataset from GMR 43-DoF retargeted motions."""

from __future__ import annotations

import pickle
from pathlib import Path

import yaml


SRC_YAML = Path("tokenhsi/data/dataset_g1_loco.yaml")
OUT_ROOT = Path("tokenhsi/data/dataset_g1_loco_body29")
OUT_YAML = Path("tokenhsi/data/dataset_g1_loco_body29.yaml")

BODY_DOF_INDICES = list(range(22)) + list(range(29, 36))
G1_BODY_LIST = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def main() -> int:
    cfg = yaml.safe_load(SRC_YAML.read_text())
    if cfg.get("format") != "gmr_robot_motion":
        raise ValueError(f"{SRC_YAML} is not a gmr_robot_motion dataset")

    out_cfg = {"format": "gmr_robot_motion", "motions": {"loco": []}}
    base_dir = SRC_YAML.parent

    for entry in cfg["motions"]["loco"]:
        src = base_dir / entry["file"]
        rel_tail = Path(*Path(entry["file"]).parts[2:])
        dst = OUT_ROOT / "motions" / rel_tail
        dst.parent.mkdir(parents=True, exist_ok=True)

        with src.open("rb") as f:
            data = pickle.load(f)

        link_body_list = data["link_body_list"]
        body_indices = [link_body_list.index(name) for name in G1_BODY_LIST]

        data = dict(data)
        data["dof_pos"] = data["dof_pos"][:, BODY_DOF_INDICES]
        data["local_body_pos"] = data["local_body_pos"][:, body_indices, :]
        data["link_body_list"] = list(G1_BODY_LIST)

        with dst.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        out_cfg["motions"]["loco"].append(
            {"file": str(Path("dataset_g1_loco_body29") / "motions" / rel_tail), "weight": entry["weight"]}
        )

    OUT_YAML.write_text(yaml.safe_dump(out_cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(out_cfg['motions']['loco'])} motions to {OUT_ROOT}")
    print(f"Wrote {OUT_YAML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
