import os
import pickle
import xml.etree.ElementTree as ET

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


def _axis_angle_to_quat_xyzw(axis, angle):
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1e-8)
    half_angle = 0.5 * angle.unsqueeze(-1)
    xyz = axis * torch.sin(half_angle)
    w = torch.cos(half_angle)
    return torch.cat((xyz, w), dim=-1)


def _quat_conjugate_xyzw(q):
    out = q.clone()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def _quat_rotate_xyzw(q, v):
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


class _MJCFKinematics:
    def __init__(self, asset_file, device):
        self._device = device
        self._body_names = []
        self._parent_ids = []
        self._local_pos = []
        self._local_rot = []
        self._joint_axes = []
        self._joint_dof_ids = []
        self._parse(asset_file)

        self._parent_ids = torch.tensor(self._parent_ids, dtype=torch.long, device=device)
        self._local_pos = torch.tensor(np.asarray(self._local_pos), dtype=torch.float32, device=device)
        self._local_rot = _normalize_quat(torch.tensor(np.asarray(self._local_rot), dtype=torch.float32, device=device))
        if self._joint_axes:
            self._joint_axes = torch.stack(self._joint_axes, dim=0).to(device=device, dtype=torch.float32)
        else:
            self._joint_axes = torch.zeros((0, 3), dtype=torch.float32, device=device)

    @property
    def body_names(self):
        return self._body_names

    @property
    def num_dof(self):
        return int(self._joint_axes.shape[0])

    def matches(self, link_body_list, dof_dim):
        return self.num_dof == dof_dim and list(link_body_list) == self._body_names

    def forward(self, root_pos, root_rot, dof_pos):
        body_pos = [None] * len(self._body_names)
        body_rot = [None] * len(self._body_names)
        body_pos[0] = root_pos
        body_rot[0] = root_rot

        for body_id in range(1, len(self._body_names)):
            parent_id = self._parent_ids[body_id].item()
            parent_pos = body_pos[parent_id]
            parent_rot = body_rot[parent_id]
            local_pos = self._local_pos[body_id].expand_as(parent_pos)
            local_rot = self._local_rot[body_id].expand_as(parent_rot)
            joint_dof_id = self._joint_dof_ids[body_id]

            if joint_dof_id is None:
                joint_rot = torch.zeros_like(parent_rot)
                joint_rot[..., 3] = 1.0
            else:
                axis = self._joint_axes[joint_dof_id].expand_as(parent_pos)
                joint_rot = _axis_angle_to_quat_xyzw(axis, dof_pos[..., joint_dof_id])

            body_pos[body_id] = parent_pos + _quat_rotate_xyzw(parent_rot, local_pos)
            body_rot[body_id] = _normalize_quat(_quat_mul_xyzw(parent_rot, _quat_mul_xyzw(local_rot, joint_rot)))

        return torch.stack(body_pos, dim=-2), torch.stack(body_rot, dim=-2)

    def _parse(self, asset_file):
        root = ET.parse(asset_file).getroot()
        compiler = root.find("compiler")
        rot_unit = "degree" if compiler is None else compiler.attrib.get("angle", "degree")
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("{} is missing worldbody".format(asset_file))

        def parse_body(body_node, parent_id):
            body_id = len(self._body_names)
            self._body_names.append(body_node.attrib["name"])
            self._parent_ids.append(parent_id)
            self._local_pos.append(np.fromstring(body_node.attrib.get("pos", "0 0 0"), dtype=np.float32, sep=" "))
            quat_wxyz = np.fromstring(body_node.attrib.get("quat", "1 0 0 0"), dtype=np.float32, sep=" ")
            self._local_rot.append(np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32))

            joints = [j for j in body_node.findall("joint") if j.attrib.get("type") != "free"]
            if len(joints) > 1:
                raise ValueError("{} body {} has {} joints; only 0/1 DOF MJCF bodies are supported".format(asset_file, self._body_names[-1], len(joints)))
            if joints:
                axis = np.fromstring(joints[0].attrib.get("axis", "0 0 1"), dtype=np.float32, sep=" ")
                if rot_unit == "degree":
                    axis = axis.astype(np.float32)
                self._joint_axes.append(torch.tensor(axis, device=self._device, dtype=torch.float32))
                self._joint_dof_ids.append(len(self._joint_axes) - 1)
            else:
                self._joint_dof_ids.append(None)

            for child in body_node.findall("body"):
                parse_body(child, body_id)

        roots = worldbody.findall("body")
        if len(roots) != 1:
            raise ValueError("{} expected one root body, got {}".format(asset_file, len(roots)))
        parse_body(roots[0], -1)


class GMRRobotMotionLib:
    def __init__(self, motion_file, skill, key_body_ids, device):
        self._device = device
        self._key_body_ids = torch.tensor(key_body_ids, device=device, dtype=torch.long)
        self._motions = []
        self._motion_weights = []
        self._motion_lengths = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._fk_model = None
        self._fk_warning_printed = False
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
        body_rot0, body_rot1 = self._gather("body_rot", motion_ids, frame_idx0, frame_idx1)
        body_pos = (1.0 - blend_body) * body_pos0 + blend_body * body_pos1

        num_bodies = body_pos.shape[1]
        body_rot = _quat_lerp(body_rot0, body_rot1, blend_body)
        body_vel = self._gather_single("body_vel", motion_ids, frame_idx0)
        body_ang_vel = self._gather_single("body_ang_vel", motion_ids, frame_idx0)
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
        self._fk_model = self._load_fk_model(cfg.get("asset_file"), dir_name)
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

        root_vel = _as_float_tensor(data["root_vel"], self._device) if "root_vel" in data else self._finite_difference(root_pos, fps)
        dof_vel = _as_float_tensor(data["dof_vel"], self._device) if "dof_vel" in data else self._finite_difference(dof_pos, fps)
        root_ang_vel = _as_float_tensor(data["root_ang_vel"], self._device) if "root_ang_vel" in data else self._quat_angular_velocity(root_rot, fps)

        if "body_pos" in data:
            body_pos = _as_float_tensor(data["body_pos"], self._device)
        else:
            body_pos = self._body_pos_from_local(root_pos, root_rot, local_body_pos)

        if "body_rot" in data:
            body_rot = _normalize_quat(_as_float_tensor(data["body_rot"], self._device))
        else:
            body_rot = self._fallback_body_rot(root_rot, body_pos.shape[1])
            if self._fk_model is not None and self._fk_model.matches(data.get("link_body_list", []), dof_pos.shape[1]):
                _, body_rot = self._fk_model.forward(root_pos, root_rot, dof_pos)
            elif not self._fk_warning_printed and body_pos.shape[1] > 0:
                print("GMRRobotMotionLib: no matching MJCF FK model; non-root body_rot/body_ang_vel use root fallback.")
                self._fk_warning_printed = True

        body_vel = _as_float_tensor(data["body_vel"], self._device) if "body_vel" in data else self._finite_difference(body_pos, fps)
        body_ang_vel = _as_float_tensor(data["body_ang_vel"], self._device) if "body_ang_vel" in data else self._quat_angular_velocity(body_rot, fps)

        return {
            "fps": fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "body_pos": body_pos,
            "body_rot": body_rot,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_vel": dof_vel,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "link_body_list": data.get("link_body_list", []),
        }

    def _finite_difference(self, values, fps):
        if values.shape[0] < 2:
            return torch.zeros_like(values)
        vel = torch.zeros_like(values)
        vel[:-1] = (values[1:] - values[:-1]) * fps
        vel[-1] = vel[-2]
        return vel

    def _quat_angular_velocity(self, quat, fps):
        if quat.shape[0] < 2:
            return torch.zeros(quat.shape[:-1] + (3,), device=self._device, dtype=torch.float32)

        q0 = quat[:-1]
        q1 = quat[1:]
        same_hemi = torch.sum(q0 * q1, dim=-1, keepdim=True) >= 0.0
        q1 = torch.where(same_hemi, q1, -q1)
        dq = _normalize_quat(_quat_mul_xyzw(q1, _quat_conjugate_xyzw(q0)))

        xyz = dq[..., :3]
        xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(xyz_norm, torch.clamp(dq[..., 3:4], min=-1.0, max=1.0))
        angle = torch.where(angle > np.pi, angle - 2.0 * np.pi, angle)
        axis = xyz / torch.clamp(xyz_norm, min=1e-8)
        frame_ang_vel = axis * angle * fps
        frame_ang_vel = torch.where(xyz_norm > 1e-8, frame_ang_vel, torch.zeros_like(frame_ang_vel))

        ang_vel = torch.zeros(quat.shape[:-1] + (3,), device=self._device, dtype=torch.float32)
        ang_vel[:-1] = frame_ang_vel
        ang_vel[-1] = ang_vel[-2]
        return ang_vel

    def _fallback_body_rot(self, root_rot, num_bodies):
        body_rot = torch.zeros((root_rot.shape[0], num_bodies, 4), device=self._device, dtype=torch.float32)
        body_rot[..., 3] = 1.0
        if num_bodies > 0:
            body_rot[:, 0, :] = root_rot
        return body_rot

    def _load_fk_model(self, asset_file, dir_name):
        candidates = []
        if asset_file:
            candidates.append(asset_file if os.path.isabs(asset_file) else os.path.join(dir_name, asset_file))
        candidates.extend(
            [
                os.path.join(os.getcwd(), "tokenhsi/data/assets/mjcf/g1_mocap_29dof_with_hands.xml"),
                os.path.join(os.getcwd(), "assets/unitree_g1/g1_mocap_29dof_with_hands.xml"),
            ]
        )
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                if os.path.splitext(candidate)[1].lower() != ".xml":
                    continue
                return _MJCFKinematics(candidate, self._device)
        return None

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
