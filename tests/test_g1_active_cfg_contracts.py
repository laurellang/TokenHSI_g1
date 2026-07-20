from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import pickle

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml"
ACTIVE_G1_ASSET = "mjcf/g1_dex3_ori.urdf"


def load_env_cfg():
    with ENV_CFG.open("r") as f:
        return yaml.safe_load(f)["env"]


def test_carry_box_sizes_are_scaled_from_humanoid_v3_to_g1_height():
    env = load_env_cfg()
    build = env["carry"]["box"]["build"]
    ratio = 0.793 / 0.94

    assert build["baseSize"] == pytest.approx([0.4 * ratio] * 3, abs=1e-6)
    assert build["scaleRangeX"] == [0.5, 1.5]
    assert build["scaleRangeY"] == [0.5, 1.5]
    assert build["scaleRangeZ"] == [0.5, 1.5]
    assert build["scaleSampleInterval"] == pytest.approx(0.125)

    source_test_sizes = [
        [0.22, 0.22, 0.22],
        [0.27, 0.27, 0.27],
        [0.32, 0.32, 0.32],
        [0.37, 0.37, 0.37],
        [0.42, 0.42, 0.42],
        [0.47, 0.47, 0.47],
        [0.52, 0.52, 0.52],
        [0.57, 0.57, 0.57],
        [0.30, 0.30, 0.40],
    ]
    expected = [[axis * ratio for axis in size] for size in source_test_sizes]
    assert len(build["testSizes"]) == len(expected)
    for actual_size, expected_size in zip(build["testSizes"], expected):
        assert actual_size == pytest.approx(expected_size, abs=1e-6)


def test_entrypoints_use_active_g1_dex3_cfg():
    cfg_ref = "tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml"
    stale_ref = "tokenhsi/data/cfg/multi_task/amp_g1_brainco_traj_sit_carry_climb.yaml"
    required_files = [
        ROOT / "tools/validate_g1_interfaces.py",
        ROOT / "tokenhsi/scripts/single_task/g1_tokenhsi_traj_train.sh",
    ]
    optional_files = [
        ROOT / "README.md",
        ROOT / "TOKENHSI_G1_REMOTE_RUNBOOK.md",
    ]

    for path in required_files:
        assert path.exists(), "{} missing".format(path)
        text = path.read_text()
        assert cfg_ref in text
        assert stale_ref not in text

    for path in optional_files:
        if not path.exists():
            continue
        text = path.read_text()
        assert stale_ref not in text


def test_active_cfg_uses_native_g1_dex3_urdf_asset():
    env = load_env_cfg()

    assert env["asset"]["assetFileName"] == ACTIVE_G1_ASSET


def test_active_cfg_masks_all_low_g1_foot_links_from_fall_termination():
    env = load_env_cfg()

    assert set(env["contactBodies"]) >= {
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    }


def test_traj_only_cfg_uses_primitive_climb_assets_for_isaacgym_smoke():
    env = load_env_cfg()

    assert env["taskInitProb"] == [1.0, 0.0, 0.0, 0.0]
    assert env["loadInactiveTaskAssets"] is False
    assert env["climb"]["objCategories"] == ["Box"]


def test_documented_smoke_command_satisfies_amp_minibatch_contract():
    train_cfg = ROOT / "tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml"
    with train_cfg.open("r") as f:
        train = yaml.safe_load(f)["params"]["config"]

    amp_minibatch_size = train["amp_minibatch_size"]
    smoke_num_envs = 64
    smoke_horizon_length = 64
    smoke_minibatch_size = 4096
    smoke_batch_size = smoke_num_envs * smoke_horizon_length

    assert amp_minibatch_size <= smoke_minibatch_size
    assert smoke_batch_size % smoke_minibatch_size == 0

    for path in [ROOT / "README.md", ROOT / "TOKENHSI_G1_REMOTE_RUNBOOK.md"]:
        if not path.exists():
            continue
        text = path.read_text()
        assert "--num_envs 64" in text
        assert "--horizon_length 64" in text
        assert "--minibatch_size 4096" in text
        assert "--minibatch_size 512" not in text


def test_task_object_assets_exist_for_active_cfg():
    env = load_env_cfg()
    data_root = ROOT / "tokenhsi/data"
    expected = {
        "sit": env["sit"]["objCategories"],
        "climb": env["climb"]["objCategories"],
    }

    for task_name, categories in expected.items():
        dataset_root = data_root / env[task_name]["objDatasetDir"]
        assert dataset_root.exists(), "{} object root missing".format(task_name)
        for mode in ["train", "test"]:
            for category in categories:
                category_dir = dataset_root / mode / category
                assert category_dir.exists(), "{} {} {} missing".format(task_name, mode, category)
                object_dirs = [p for p in category_dir.iterdir() if p.is_dir()]
                assert object_dirs, "{} {} {} has no objects".format(task_name, mode, category)
                first_obj = object_dirs[0]
                assert (first_obj / "asset.urdf").exists()
                assert (first_obj / "config.json").exists()


def test_active_g1_urdf_has_43_actuated_joints_and_resolved_meshes():
    env = load_env_cfg()
    urdf_path = ROOT / "tokenhsi/data/assets" / env["asset"]["assetFileName"]
    root = ET.parse(urdf_path).getroot()

    actuated_joints = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]
    mesh_paths = [mesh.attrib["filename"] for mesh in root.findall(".//mesh")]
    missing_meshes = [
        mesh_path
        for mesh_path in mesh_paths
        if not (urdf_path.parent / mesh_path).exists()
    ]

    assert len(actuated_joints) == 43
    assert actuated_joints[0] == "left_hip_pitch_joint"
    assert actuated_joints[-1] == "right_hand_index_1_joint"
    assert missing_meshes == []


def test_active_g1_urdf_dof_order_and_ranges_match_retarget_reference_mjcf():
    env = load_env_cfg()
    urdf_path = ROOT / "tokenhsi/data/assets" / env["asset"]["assetFileName"]
    mjcf_path = ROOT / "tokenhsi/data/assets/mjcf/g1_mocap_29dof_with_hands.xml"

    urdf_root = ET.parse(urdf_path).getroot()
    mjcf_root = ET.parse(mjcf_path).getroot()

    urdf_rows = []
    for joint in urdf_root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        child = joint.find("child")
        urdf_rows.append(
            {
                "name": joint.attrib["name"],
                "child": child.attrib["link"],
                "lower": float(limit.attrib["lower"]),
                "upper": float(limit.attrib["upper"]),
            }
        )

    mjcf_rows = []

    def walk_mjcf_body(body):
        for joint in body.findall("joint"):
            if joint.attrib.get("name") is not None and joint.attrib.get("type") != "free":
                lower, upper = joint.attrib["range"].split()
                mjcf_rows.append(
                    {
                        "name": joint.attrib["name"],
                        "child": body.attrib["name"],
                        "lower": float(lower),
                        "upper": float(upper),
                    }
                )
        for child_body in body.findall("body"):
            walk_mjcf_body(child_body)

    for body in mjcf_root.find("worldbody").findall("body"):
        walk_mjcf_body(body)

    assert len(urdf_rows) == 43
    assert len(mjcf_rows) == 43
    assert [row["name"] for row in urdf_rows] == [row["name"] for row in mjcf_rows]
    for urdf_row, mjcf_row in zip(urdf_rows, mjcf_rows):
        assert urdf_row["lower"] == pytest.approx(mjcf_row["lower"], abs=1e-5)
        assert urdf_row["upper"] == pytest.approx(mjcf_row["upper"], abs=1e-5)

    assert [row["name"] for row in urdf_rows[22:29]] == [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ]
    assert [row["name"] for row in urdf_rows[36:43]] == [
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ]


def test_validate_g1_interfaces_checks_dof_alignment_contract():
    validator = ROOT / "tools/validate_g1_interfaces.py"
    text = validator.read_text()

    assert "validate_dof_alignment" in text
    assert "g1_mocap_29dof_with_hands.xml" in text
    assert "DOF order mismatch" in text


def test_active_g1_native_urdf_body_list_is_mappable_to_retargeted_motion_schema():
    env = load_env_cfg()
    urdf_path = ROOT / "tokenhsi/data/assets" / env["asset"]["assetFileName"]
    motion_path = next((ROOT / "tokenhsi/data/dataset_g1_all").rglob("*.pkl"))
    root = ET.parse(urdf_path).getroot()

    with motion_path.open("rb") as f:
        motion = pickle.load(f)

    urdf_links = [link.attrib["name"] for link in root.findall("link")]
    motion_bodies = list(motion["link_body_list"])

    assert len(urdf_links) == 53
    assert len(motion_bodies) == 52
    assert "pelvis_contour_link" in urdf_links
    assert "left_toe_link" in motion_bodies
    assert "left_toe_link" not in urdf_links
    for name in load_env_cfg()["keyBodies"]:
        assert name in urdf_links
        assert name in motion_bodies


def test_humanoid_loader_accepts_native_g1_dex3_urdf_asset():
    env = load_env_cfg()
    humanoid_py = ROOT / "tokenhsi/env/tasks/humanoid.py"
    text = humanoid_py.read_text()

    assert env["asset"]["assetFileName"] in text



def test_g1_termination_height_uses_g1_head_link_name():
    humanoid_py = ROOT / "tokenhsi/env/tasks/humanoid.py"
    text = humanoid_py.read_text()

    assert "head_link" in text
    assert "head body" in text

def test_humanoid_loader_keeps_gmr_asset_collision_geometry():
    humanoid_py = ROOT / "tokenhsi/env/tasks/humanoid.py"
    text = humanoid_py.read_text()

    assert "asset_options.replace_cylinder_with_capsule = False" in text


def test_humanoid_loader_falls_back_to_urdf_dof_effort_when_actuators_are_missing():
    humanoid_py = ROOT / "tokenhsi/env/tasks/humanoid.py"
    text = humanoid_py.read_text()

    assert "if len(motor_efforts) == 0:" in text
    assert "dof_prop_for_effort = self.gym.get_asset_dof_properties(humanoid_asset)" in text
    assert "motor_efforts = dof_prop_for_effort['effort'].tolist()" in text
    assert "len(motor_efforts) != self.num_dof" in text


def test_active_g1_cfg_declares_unitree_motor_control_contract():
    env = load_env_cfg()
    control = env["g1Control"]

    assert control["useUnitreeMotorConstants"] is True
    assert control["useDefaultJointAngles"] is True
    assert control["actionScaleFactor"] == pytest.approx(0.25)
    assert control["terminationGraceSteps"] >= 10
    assert control["handKp"] > 0.0
    assert control["handKd"] > 0.0
    assert control["handActionScale"] > 0.0


def test_humanoid_loader_applies_g1_motor_gains_and_default_action_offset():
    humanoid_py = ROOT / "tokenhsi/env/tasks/humanoid.py"
    text = humanoid_py.read_text()

    assert "_apply_g1_pd_control_props" in text
    assert "_build_g1_motor_pd_tables" in text
    assert 'dof_prop["stiffness"]' in text
    assert 'dof_prop["damping"]' in text
    assert 'dof_prop["armature"]' in text
    assert "_build_g1_default_dof_pos" in text
    assert "useDefaultJointAngles" in text



def test_skeleton_motion_import_does_not_warn_when_fbx_is_missing():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lpanlib.poselib.skeleton.skeleton3d",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    combined_output = result.stdout + result.stderr
    assert "FBX library failed to load" not in combined_output
