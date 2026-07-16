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


if __name__ == "__main__":
    unittest.main()
