import pickle
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenhsi.utils.gmr_robot_motion_lib import GMRRobotMotionLib


def _yaw_quat(yaw):
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)], dtype=np.float32)


class GMRRobotMotionLibTest(unittest.TestCase):
    def test_interpolates_robot_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            motion_path = tmp_path / "walk.pkl"
            motion_data = {
                "fps": 2,
                "root_pos": np.array([[0.0, 0.0, 0.8], [1.0, 0.0, 0.8]], dtype=np.float32),
                "root_rot": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                "dof_pos": np.array([[0.0, 0.2, -0.2], [1.0, 1.2, 0.8]], dtype=np.float32),
                "local_body_pos": np.array(
                    [
                        [[0.0, 0.0, 0.0], [0.2, 0.0, -0.7], [-0.2, 0.0, -0.7]],
                        [[0.0, 0.0, 0.0], [0.2, 0.0, -0.7], [-0.2, 0.0, -0.7]],
                    ],
                    dtype=np.float32,
                ),
                "link_body_list": ["pelvis", "right_ankle_roll_link", "left_ankle_roll_link"],
            }
            with motion_path.open("wb") as f:
                pickle.dump(motion_data, f)

            motion_yaml = tmp_path / "dataset_g1_loco.yaml"
            motion_yaml.write_text(
                yaml.safe_dump(
                    {
                        "format": "gmr_robot_motion",
                        "motions": {
                            "loco": [
                                {
                                    "file": motion_path.name,
                                    "weight": 1.0,
                                }
                            ]
                        },
                    }
                )
            )

            lib = GMRRobotMotionLib(
                motion_file=str(motion_yaml),
                skill="loco",
                key_body_ids=[1, 2],
                device="cpu",
            )

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = lib.get_motion_state(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0.25], dtype=torch.float32),
            )

            self.assertTrue(torch.allclose(root_pos, torch.tensor([[0.5, 0.0, 0.8]])))
            self.assertTrue(torch.allclose(root_rot, torch.tensor([[0.0, 0.0, 0.0, 1.0]])))
            self.assertTrue(torch.allclose(dof_pos, torch.tensor([[0.5, 0.7, 0.3]]), atol=1e-6))
            self.assertTrue(torch.allclose(root_vel, torch.tensor([[2.0, 0.0, 0.0]]), atol=1e-6))
            self.assertTrue(torch.allclose(root_ang_vel, torch.zeros(1, 3), atol=1e-6))
            self.assertTrue(torch.allclose(dof_vel, torch.tensor([[2.0, 2.0, 2.0]]), atol=1e-6))
            self.assertTrue(
                torch.allclose(
                    key_pos,
                    torch.tensor([[[0.7, 0.0, 0.1], [0.3, 0.0, 0.1]]]),
                    atol=1e-6,
                )
            )

    def test_derives_root_angular_velocity_from_xyzw_quaternion_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            motion_path = tmp_path / "turn.pkl"
            motion_data = {
                "fps": 2,
                "root_pos": np.zeros((3, 3), dtype=np.float32),
                "root_rot": np.stack([_yaw_quat(0.0), _yaw_quat(0.5), _yaw_quat(1.0)]),
                "dof_pos": np.zeros((3, 3), dtype=np.float32),
                "local_body_pos": np.zeros((3, 1, 3), dtype=np.float32),
                "link_body_list": ["pelvis"],
            }
            with motion_path.open("wb") as f:
                pickle.dump(motion_data, f)

            motion_yaml = tmp_path / "dataset_g1_loco.yaml"
            motion_yaml.write_text(
                yaml.safe_dump(
                    {
                        "format": "gmr_robot_motion",
                        "motions": {"loco": [{"file": motion_path.name, "weight": 1.0}]},
                    }
                )
            )

            lib = GMRRobotMotionLib(
                motion_file=str(motion_yaml),
                skill="loco",
                key_body_ids=[],
                device="cpu",
            )
            _, _, _, _, root_ang_vel, _, _ = lib.get_motion_state(
                torch.tensor([0, 0], dtype=torch.long),
                torch.tensor([0.0, 0.5], dtype=torch.float32),
            )

            expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
            self.assertTrue(torch.allclose(root_ang_vel, expected, atol=1e-5), root_ang_vel)

    def test_derives_body_velocity_from_world_body_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            motion_path = tmp_path / "swing.pkl"
            motion_data = {
                "fps": 2,
                "root_pos": np.zeros((2, 3), dtype=np.float32),
                "root_rot": np.stack([_yaw_quat(0.0), _yaw_quat(np.pi / 2.0)]),
                "dof_pos": np.zeros((2, 3), dtype=np.float32),
                "local_body_pos": np.array(
                    [
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                "link_body_list": ["pelvis", "hand"],
            }
            with motion_path.open("wb") as f:
                pickle.dump(motion_data, f)

            motion_yaml = tmp_path / "dataset_g1_loco.yaml"
            motion_yaml.write_text(
                yaml.safe_dump(
                    {
                        "format": "gmr_robot_motion",
                        "motions": {"loco": [{"file": motion_path.name, "weight": 1.0}]},
                    }
                )
            )

            lib = GMRRobotMotionLib(
                motion_file=str(motion_yaml),
                skill="loco",
                key_body_ids=[1],
                device="cpu",
            )
            body_pos, _, body_vel, _ = lib.get_motion_state_max(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0.0], dtype=torch.float32),
            )

            self.assertTrue(torch.allclose(body_pos[:, 1, :], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6))
            self.assertTrue(torch.allclose(body_vel[:, 1, :], torch.tensor([[-2.0, 2.0, 0.0]]), atol=1e-5), body_vel)

    def test_derives_body_rotation_and_angular_velocity_from_mjcf_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            asset_path = tmp_path / "two_body.xml"
            asset_path.write_text(
                """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis">
      <freejoint name="root"/>
      <body name="arm" pos="1 0 0">
        <joint name="arm_joint" axis="0 0 1" range="-6.28 6.28"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
            )
            motion_path = tmp_path / "arm.pkl"
            motion_data = {
                "fps": 2,
                "root_pos": np.zeros((2, 3), dtype=np.float32),
                "root_rot": np.stack([_yaw_quat(0.0), _yaw_quat(0.0)]),
                "dof_pos": np.array([[0.0], [np.pi / 2.0]], dtype=np.float32),
                "local_body_pos": np.array(
                    [
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                "link_body_list": ["pelvis", "arm"],
            }
            with motion_path.open("wb") as f:
                pickle.dump(motion_data, f)

            motion_yaml = tmp_path / "dataset_g1_loco.yaml"
            motion_yaml.write_text(
                yaml.safe_dump(
                    {
                        "format": "gmr_robot_motion",
                        "asset_file": asset_path.name,
                        "motions": {"loco": [{"file": motion_path.name, "weight": 1.0}]},
                    }
                )
            )

            lib = GMRRobotMotionLib(
                motion_file=str(motion_yaml),
                skill="loco",
                key_body_ids=[1],
                device="cpu",
            )
            _, body_rot, _, body_ang_vel = lib.get_motion_state_max(
                torch.tensor([0], dtype=torch.long),
                torch.tensor([0.0], dtype=torch.float32),
            )

            self.assertTrue(torch.allclose(body_rot[:, 0, :], torch.tensor([[0.0, 0.0, 0.0, 1.0]]), atol=1e-6))
            self.assertTrue(torch.allclose(body_rot[:, 1, :], torch.tensor([[0.0, 0.0, 0.0, 1.0]]), atol=1e-6))
            self.assertTrue(
                torch.allclose(body_ang_vel[:, 1, :], torch.tensor([[0.0, 0.0, np.pi]]), atol=1e-5),
                body_ang_vel,
            )


if __name__ == "__main__":
    unittest.main()
