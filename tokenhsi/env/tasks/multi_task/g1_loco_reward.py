import torch
from torch import Tensor


@torch.jit.script
def _safe_normalize_xy(vec):
    # type: (Tensor) -> Tensor
    norm = torch.norm(vec, p=2, dim=-1, keepdim=True)
    return vec / torch.clamp(norm, min=1.0e-6)


@torch.jit.script
def _wrap_to_pi(angle):
    # type: (Tensor) -> Tensor
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@torch.jit.script
def _quat_heading_xyzw(q):
    # type: (Tensor) -> Tensor
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


@torch.jit.script
def compute_g1_loco_reward(root_pos, root_rot, root_vel, root_ang_vel,
                           dof_pos, dof_vel, dof_limits_lower, dof_limits_upper,
                           rigid_body_pos, rigid_body_vel, contact_forces, feet_ids,
                           actions, prev_actions, tar_pos, next_tar_pos, dt,
                           pos_weight, track_lin_vel_weight, track_ang_vel_weight, alive_weight,
                           lin_vel_z_penalty_weight, ang_vel_xy_penalty_weight,
                           flat_orientation_penalty_weight, base_height_penalty_weight,
                           base_height_target, joint_vel_penalty_weight,
                           action_rate_penalty_weight, dof_pos_limits_penalty_weight,
                           feet_slide_penalty_weight, feet_clearance_weight,
                           feet_clearance_target, feet_clearance_std, target_speed):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float) -> Tensor
    pos_diff_xy = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff_xy * pos_diff_xy, dim=-1)
    pos_reward = torch.exp(-2.0 * pos_err)

    traj_vel_xy = (next_tar_pos[..., 0:2] - tar_pos[..., 0:2]) / dt
    traj_speed = torch.norm(traj_vel_xy, p=2, dim=-1, keepdim=True)
    fallback_vel_xy = _safe_normalize_xy(pos_diff_xy) * target_speed
    target_vel_xy = torch.where(traj_speed > 0.05, traj_vel_xy, fallback_vel_xy)

    lin_vel_err = torch.sum((target_vel_xy - root_vel[..., 0:2]) ** 2, dim=-1)
    track_lin_vel_reward = torch.exp(-lin_vel_err / 0.25)

    target_heading = torch.atan2(target_vel_xy[..., 1], target_vel_xy[..., 0])
    heading = _quat_heading_xyzw(root_rot)
    heading_err = _wrap_to_pi(target_heading - heading)
    target_ang_vel_z = torch.clamp(heading_err / 0.5, -1.5, 1.5)
    ang_vel_err = (target_ang_vel_z - root_ang_vel[..., 2]) ** 2
    track_ang_vel_reward = torch.exp(-ang_vel_err / 0.25)

    lin_vel_z_penalty = root_vel[..., 2] ** 2
    ang_vel_xy_penalty = torch.sum(root_ang_vel[..., 0:2] ** 2, dim=-1)

    # For xyzw quaternions, x/y components dominate roll/pitch tilt near upright.
    flat_orientation_penalty = torch.sum(root_rot[..., 0:2] ** 2, dim=-1)
    base_height_penalty = (root_pos[..., 2] - base_height_target) ** 2

    joint_vel_penalty = torch.mean(dof_vel ** 2, dim=-1)
    action_rate_penalty = torch.mean((actions - prev_actions) ** 2, dim=-1)

    lower_violation = torch.clamp(dof_limits_lower.unsqueeze(0) - dof_pos, min=0.0)
    upper_violation = torch.clamp(dof_pos - dof_limits_upper.unsqueeze(0), min=0.0)
    dof_pos_limits_penalty = torch.sum(lower_violation + upper_violation, dim=-1)

    feet_vel_xy = rigid_body_vel[:, feet_ids, 0:2]
    feet_speed_xy = torch.norm(feet_vel_xy, p=2, dim=-1)
    feet_contact = contact_forces[:, feet_ids, 2] > 1.0
    feet_slide_penalty = torch.sum(feet_speed_xy * feet_contact.float(), dim=-1)

    feet_height = rigid_body_pos[:, feet_ids, 2]
    clearance_err = (feet_height - feet_clearance_target) ** 2
    feet_clearance_reward = torch.sum(
        torch.exp(-clearance_err / (feet_clearance_std * feet_clearance_std)) * feet_speed_xy,
        dim=-1,
    )

    reward = (
        pos_weight * pos_reward
        + track_lin_vel_weight * track_lin_vel_reward
        + track_ang_vel_weight * track_ang_vel_reward
        + alive_weight
        + lin_vel_z_penalty_weight * lin_vel_z_penalty
        + ang_vel_xy_penalty_weight * ang_vel_xy_penalty
        + flat_orientation_penalty_weight * flat_orientation_penalty
        + base_height_penalty_weight * base_height_penalty
        + joint_vel_penalty_weight * joint_vel_penalty
        + action_rate_penalty_weight * action_rate_penalty
        + dof_pos_limits_penalty_weight * dof_pos_limits_penalty
        + feet_slide_penalty_weight * feet_slide_penalty
        + feet_clearance_weight * feet_clearance_reward
    )

    return reward
