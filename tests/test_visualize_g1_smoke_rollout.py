from pathlib import Path

import numpy as np
import pytest

from tools import visualize_g1_smoke_rollout as viz


def test_load_rollout_npz_validates_names_and_pose_shapes(tmp_path: Path):
    path = tmp_path / "rollout.npz"
    np.savez(
        path,
        body_names=np.array(["torso_link", "head_link"]),
        body_pos=np.zeros((3, 2, 3), dtype=np.float32),
        body_quat_xyzw=np.zeros((3, 2, 4), dtype=np.float32),
        dt=np.array(1.0 / 60.0, dtype=np.float32),
    )

    rollout = viz.load_rollout(path)

    assert rollout.body_names == ("torso_link", "head_link")
    assert rollout.body_pos.shape == (3, 2, 3)
    assert rollout.body_quat_xyzw.shape == (3, 2, 4)
    assert rollout.dt == pytest.approx(1.0 / 60.0)


def test_load_rollout_npz_rejects_mismatched_body_count(tmp_path: Path):
    path = tmp_path / "bad_rollout.npz"
    np.savez(
        path,
        body_names=np.array(["torso_link", "head_link"]),
        body_pos=np.zeros((3, 1, 3), dtype=np.float32),
        body_quat_xyzw=np.zeros((3, 1, 4), dtype=np.float32),
        dt=np.array(1.0 / 60.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="body_names"):
        viz.load_rollout(path)


def test_make_visual_pose_applies_link_pose_and_visual_origin():
    link_pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    link_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    visual_transform = np.eye(4, dtype=np.float64)
    visual_transform[:3, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    pos, wxyz = viz.make_visual_pose(link_pos, link_quat_xyzw, visual_transform)

    np.testing.assert_allclose(pos, [1.1, 2.2, 3.3])
    np.testing.assert_allclose(wxyz, [1.0, 0.0, 0.0, 0.0])


def test_extract_rigid_body_pose_arrays_from_isaacgym_structured_state():
    dtype = np.dtype(
        [
            (
                "pose",
                [
                    ("p", [("x", "f4"), ("y", "f4"), ("z", "f4")]),
                    ("r", [("x", "f4"), ("y", "f4"), ("z", "f4"), ("w", "f4")]),
                ],
            )
        ]
    )
    state = np.zeros(2, dtype=dtype)
    state["pose"]["p"]["x"] = [1.0, 4.0]
    state["pose"]["p"]["y"] = [2.0, 5.0]
    state["pose"]["p"]["z"] = [3.0, 6.0]
    state["pose"]["r"]["x"] = [0.1, 0.5]
    state["pose"]["r"]["y"] = [0.2, 0.6]
    state["pose"]["r"]["z"] = [0.3, 0.7]
    state["pose"]["r"]["w"] = [0.4, 0.8]

    pos, quat = viz.extract_rigid_body_pose_arrays(state)

    np.testing.assert_allclose(pos, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(quat, [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])




def test_make_visual_pose_converts_isaacgym_body_frame_back_to_link_frame():
    body_pos = np.array([1.0, 2.0, 3.45], dtype=np.float64)
    body_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    inertial_transform = np.eye(4, dtype=np.float64)
    inertial_transform[:3, 3] = np.array([0.0, 0.0, 0.45], dtype=np.float64)
    visual_transform = np.eye(4, dtype=np.float64)

    pos, wxyz = viz.make_visual_pose(body_pos, body_quat_xyzw, visual_transform, inertial_transform=inertial_transform)

    np.testing.assert_allclose(pos, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(wxyz, [1.0, 0.0, 0.0, 0.0])
