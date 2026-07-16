import os
import pickle

import numpy as np
import torch
import yaml


def is_gmr_robot_motion_file(motion_file):
    if os.path.splitext(motion_file)[1] != ".yaml":
        return False
    with open(os.path.join(os.getcwd(), motion_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg.get("format") == "gmr_robot_motion"


def _load_motion_dict(path):
    ext = os.path.splitext(path)[1]
    if ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if len(data.files) == 1 and data.files[0] == "arr_0":
            return data["arr_0"].item()
        return {k: data[k] for k in data.files}
    raise ValueError("Unsupported GMR robot motion file: {}".format(path))


def _as_float_tensor(value, device):
    return torch.tensor(np.asarray(value), device=device, dtype=torch.float32)


def _normalize_quat(q):
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def _quat_lerp(q0, q1, blend):
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    q = (1.0 - blend) * q0 + blend * q1
    return _normalize_quat(q)


def _quat_rotate_xyzw(q, v):
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


class GMRRobotMotionLib:
    def __init__(self, motion_file, skill, key_body_ids, device):
        self._device = device
        self._key_body_ids = torch.tensor(key_body_ids, device=device, dtype=torch.long)
        self._motions = []
        self._motion_weights = []
        self._motion_lengths = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._load_motions(motion_file, skill)

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()

    def sample_motions(self, n):
        return torch.multinomial(self._motion_weights, num_samples=n, replacement=True)

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            motion_len = motion_len - truncate_time
        return phase * motion_len

    def sample_time_rsi(self, motion_ids, truncate_time=None):
        return self.sample_time(motion_ids, truncate_time=truncate_time)

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times):
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        blend_vec = blend.unsqueeze(-1)

        root_pos0, root_pos1 = self._gather("root_pos", motion_ids, frame_idx0, frame_idx1)
        root_rot0, root_rot1 = self._gather("root_rot", motion_ids, frame_idx0, frame_idx1)
        dof_pos0, dof_pos1 = self._gather("dof_pos", motion_ids, frame_idx0, frame_idx1)

        root_pos = (1.0 - blend_vec) * root_pos0 + blend_vec * root_pos1
        root_rot = _quat_lerp(root_rot0, root_rot1, blend_vec)
        dof_pos = (1.0 - blend_vec) * dof_pos0 + blend_vec * dof_pos1

        root_vel = self._gather_single("root_vel", motion_ids, frame_idx0)
        root_ang_vel = self._gather_single("root_ang_vel", motion_ids, frame_idx0)
        dof_vel = self._gather_single("dof_vel", motion_ids, frame_idx0)
        key_pos = self._compute_key_pos(motion_ids, frame_idx0, frame_idx1, blend, root_pos, root_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos

    def get_motion_state_max(self, motion_ids, motion_times):
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        blend_body = blend[:, None, None]
        root_pos, root_rot, _, root_vel, root_ang_vel, _, _ = self.get_motion_state(motion_ids, motion_times)
        body_pos0, body_pos1 = self._gather("body_pos", motion_ids, frame_idx0, frame_idx1)
        body_pos = (1.0 - blend_body) * body_pos0 + blend_body * body_pos1

        num_bodies = body_pos.shape[1]
        body_rot = torch.zeros((body_pos.shape[0], num_bodies, 4), device=self._device, dtype=torch.float32)
        body_rot[..., 3] = 1.0
        if num_bodies > 0:
            body_rot[:, 0, :] = root_rot
        body_vel = root_vel[:, None, :].repeat(1, num_bodies, 1)
        body_ang_vel = root_ang_vel[:, None, :].repeat(1, num_bodies, 1)
        return body_pos, body_rot, body_vel, body_ang_vel

    def get_obj_motion_state(self, motion_ids, motion_times):
        raise NotImplementedError("GMR robot loco motions do not include object motion.")

    def get_obj_motion_state_single_frame(self, motion_ids):
        raise NotImplementedError("GMR robot loco motions do not include object motion.")

    def _load_motions(self, motion_file, skill):
        with open(os.path.join(os.getcwd(), motion_file), "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        if cfg.get("format") != "gmr_robot_motion":
            raise ValueError("Expected format: gmr_robot_motion in {}".format(motion_file))

        dir_name = os.path.dirname(motion_file)
        motion_entries = cfg["motions"][skill]
        for entry in motion_entries:
            curr_file = os.path.join(dir_name, entry["file"])
            data = _load_motion_dict(curr_file)
            motion = self._prepare_motion(data, curr_file)
            self._motions.append(motion)
            self._motion_weights.append(float(entry["weight"]))
            self._motion_lengths.append((motion["root_pos"].shape[0] - 1) / motion["fps"])
            self._motion_dt.append(1.0 / motion["fps"])
            self._motion_num_frames.append(motion["root_pos"].shape[0])

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)

    def _prepare_motion(self, data, path):
        required = ["fps", "root_pos", "root_rot", "dof_pos"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError("{} is missing keys: {}".format(path, ", ".join(missing)))

        root_pos = _as_float_tensor(data["root_pos"], self._device)
        root_rot = _normalize_quat(_as_float_tensor(data["root_rot"], self._device))
        dof_pos = _as_float_tensor(data["dof_pos"], self._device)
        fps = float(np.asarray(data["fps"]).item())

        if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
            raise ValueError("{} has inconsistent frame counts".format(path))

        local_body_pos = data.get("local_body_pos")
        if local_body_pos is None:
            local_body_pos = np.zeros((root_pos.shape[0], 0, 3), dtype=np.float32)
        local_body_pos = _as_float_tensor(local_body_pos, self._device)
        if local_body_pos.shape[0] != root_pos.shape[0]:
            raise ValueError("{} local_body_pos frame count does not match root_pos".format(path))

        root_vel = self._finite_difference(root_pos, fps)
        dof_vel = self._finite_difference(dof_pos, fps)
        root_ang_vel = torch.zeros((root_pos.shape[0], 3), device=self._device, dtype=torch.float32)
        body_pos = self._body_pos_from_local(root_pos, root_rot, local_body_pos)

        return {
            "fps": fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "body_pos": body_pos,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel": dof_vel,
            "link_body_list": data.get("link_body_list", []),
        }

    def _finite_difference(self, values, fps):
        if values.shape[0] < 2:
            return torch.zeros_like(values)
        vel = torch.zeros_like(values)
        vel[:-1] = (values[1:] - values[:-1]) * fps
        vel[-1] = vel[-2]
        return vel

    def _body_pos_from_local(self, root_pos, root_rot, local_body_pos):
        if local_body_pos.shape[1] == 0:
            return local_body_pos
        root_rot_exp = root_rot[:, None, :].expand(-1, local_body_pos.shape[1], -1)
        return root_pos[:, None, :] + _quat_rotate_xyzw(root_rot_exp, local_body_pos)

    def _calc_frame_blend(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        phase = torch.clip(motion_times / motion_len, 0.0, 1.0)
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (motion_times - frame_idx0 * dt) / dt
        blend = torch.clip(blend, 0.0, 1.0)
        return frame_idx0, frame_idx1, blend

    def _gather(self, key, motion_ids, frame_idx0, frame_idx1):
        return (
            self._gather_single(key, motion_ids, frame_idx0),
            self._gather_single(key, motion_ids, frame_idx1),
        )

    def _gather_single(self, key, motion_ids, frame_idx):
        values = []
        for motion_id, frame_id in zip(motion_ids.tolist(), frame_idx.tolist()):
            values.append(self._motions[motion_id][key][frame_id])
        return torch.stack(values, dim=0)

    def _compute_key_pos(self, motion_ids, frame_idx0, frame_idx1, blend, root_pos, root_rot):
        if self._key_body_ids.numel() == 0:
            return torch.zeros((motion_ids.shape[0], 0, 3), device=self._device, dtype=torch.float32)

        body_pos0, body_pos1 = self._gather("body_pos", motion_ids, frame_idx0, frame_idx1)
        blend_exp = blend[:, None, None]
        body_pos = (1.0 - blend_exp) * body_pos0 + blend_exp * body_pos1
        return body_pos[:, self._key_body_ids, :]
