from pathlib import Path

import pytest

from tools import visualize_g1_brainco_urdf as viz


def test_default_urdf_path_points_to_brainco_asset():
    args = viz.parse_args([])

    assert args.urdf == Path("assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf")
    assert viz.resolve_urdf_path(args.urdf).exists()


def test_load_urdf_model_finds_joints_and_visuals():
    model = viz.load_urdf_model(viz.DEFAULT_URDF)

    assert model.root_link == "pelvis"
    assert len(model.joints) >= 29
    assert len(model.visuals_by_link["pelvis"]) >= 1

    joints_by_name = {joint.name: joint for joint in model.joints}
    assert joints_by_name["left_hip_roll_joint"].axis == pytest.approx((1.0, 0.0, 0.0))
    assert joints_by_name["left_hip_yaw_joint"].axis == pytest.approx((0.0, 0.0, 1.0))


def test_joint_slider_range_uses_limits_or_default():
    limited = viz.Joint(
        name="limited",
        joint_type="revolute",
        parent="a",
        child="b",
        origin=viz.Transform.identity(),
        axis=(0.0, 0.0, 1.0),
        limit_lower=-0.5,
        limit_upper=0.75,
    )
    continuous = viz.Joint(
        name="continuous",
        joint_type="continuous",
        parent="a",
        child="b",
        origin=viz.Transform.identity(),
        axis=(0.0, 0.0, 1.0),
        limit_lower=None,
        limit_upper=None,
    )

    assert viz.slider_range(limited) == pytest.approx((-0.5, 0.75))
    assert viz.slider_range(continuous) == pytest.approx((-3.141592653589793, 3.141592653589793))
