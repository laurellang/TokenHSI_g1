from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml"


def test_traj_cfg_declares_g1_loco_reward_terms():
    with ENV_CFG.open("r") as f:
        env = yaml.safe_load(f)["env"]

    reward = env["traj"]["reward"]
    required_terms = {
        "posWeight",
        "trackLinVelWeight",
        "trackAngVelWeight",
        "aliveWeight",
        "linVelZPenaltyWeight",
        "angVelXYPenaltyWeight",
        "flatOrientationPenaltyWeight",
        "baseHeightPenaltyWeight",
        "jointVelPenaltyWeight",
        "actionRatePenaltyWeight",
        "dofPosLimitsPenaltyWeight",
        "feetSlidePenaltyWeight",
        "feetClearanceWeight",
    }

    assert required_terms <= set(reward)
    assert reward["baseHeightTarget"] == pytest.approx(0.78, abs=1e-6)


def test_g1_loco_reward_prefers_tracking_the_trajectory_velocity():
    from tokenhsi.env.tasks.multi_task.g1_loco_reward import compute_g1_loco_reward

    n = 2
    root_pos = torch.tensor([[0.0, 0.0, 0.78], [0.0, 0.0, 0.78]], dtype=torch.float32)
    root_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    root_vel = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    root_ang_vel = torch.zeros((n, 3), dtype=torch.float32)
    dof_pos = torch.zeros((n, 43), dtype=torch.float32)
    dof_vel = torch.zeros_like(dof_pos)
    dof_limits_lower = torch.full((43,), -1.0, dtype=torch.float32)
    dof_limits_upper = torch.full((43,), 1.0, dtype=torch.float32)
    rigid_body_pos = torch.zeros((n, 4, 3), dtype=torch.float32)
    rigid_body_pos[:, :, 2] = 0.05
    rigid_body_vel = torch.zeros_like(rigid_body_pos)
    contact_forces = torch.zeros_like(rigid_body_pos)
    feet_ids = torch.tensor([2, 3], dtype=torch.long)
    actions = torch.zeros((n, 43), dtype=torch.float32)
    prev_actions = torch.zeros_like(actions)
    tar_pos = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    next_tar_pos = torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=torch.float32)

    reward = compute_g1_loco_reward(
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        dof_limits_lower,
        dof_limits_upper,
        rigid_body_pos,
        rigid_body_vel,
        contact_forces,
        feet_ids,
        actions,
        prev_actions,
        tar_pos,
        next_tar_pos,
        0.1,
        0.3,
        1.0,
        0.5,
        0.15,
        -2.0,
        -0.05,
        -5.0,
        -10.0,
        0.78,
        -0.001,
        -0.05,
        -5.0,
        -0.2,
        1.0,
        0.10,
        0.05,
        1.0,
    )

    assert reward[0] > reward[1]
