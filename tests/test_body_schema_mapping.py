from pathlib import Path

import torch

from tokenhsi.utils.body_schema import build_asset_to_motion_body_mapping, map_motion_body_state_to_asset

ROOT = Path(__file__).resolve().parents[1]


def test_missing_fixed_asset_bodies_keep_parent_motion_plus_rest_offset():
    asset_body_names = [
        "pelvis",
        "torso_link",
        "head_link",
        "left_wrist_yaw_link",
        "left_hand_palm_link",
    ]
    asset_body_parent_names = {
        "pelvis": None,
        "torso_link": "pelvis",
        "head_link": "torso_link",
        "left_wrist_yaw_link": "torso_link",
        "left_hand_palm_link": "left_wrist_yaw_link",
    }
    asset_body_rest_pos = {
        "pelvis": (0.0, 0.0, 0.0),
        "torso_link": (0.0, 0.0, 0.2),
        "head_link": (0.0039635, 0.0, 0.156),
        "left_wrist_yaw_link": (0.4, 0.2, 0.5),
        "left_hand_palm_link": (0.4415, 0.203, 0.5),
    }
    motion_body_names = ["pelvis", "torso_link", "left_wrist_yaw_link"]

    body_ids, pos_offsets = build_asset_to_motion_body_mapping(
        asset_body_names,
        asset_body_parent_names,
        motion_body_names,
        asset_body_rest_pos,
    )

    assert body_ids.tolist() == [0, 1, 1, 2, 2]
    assert torch.allclose(pos_offsets[2], torch.tensor([0.0039635, 0.0, -0.044]))
    assert torch.allclose(pos_offsets[4], torch.tensor([0.0415, 0.003, 0.0]))

    motion_state = torch.zeros((1, 3, 13), dtype=torch.float32)
    motion_state[0, 1, 0:3] = torch.tensor([1.0, 2.0, 3.0])
    motion_state[0, 2, 0:3] = torch.tensor([4.0, 5.0, 6.0])
    motion_state[0, :, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    motion_state[0, 2, 7:10] = torch.tensor([0.1, 0.2, 0.3])

    asset_state = map_motion_body_state_to_asset(motion_state, body_ids, pos_offsets)

    assert torch.allclose(asset_state[0, 1, 0:3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(asset_state[0, 2, 0:3], torch.tensor([1.0039635, 2.0, 2.956]))
    assert torch.allclose(asset_state[0, 4, 0:3], torch.tensor([4.0415, 5.003, 6.0]))
    assert torch.allclose(asset_state[0, 4, 7:10], torch.tensor([0.1, 0.2, 0.3]))


def test_humanoid_motion_body_mapping_does_not_skip_when_body_counts_match():
    task_source = ROOT / "tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py"
    text = task_source.read_text()

    assert "or body_state.shape[1] == self.num_bodies" not in text
