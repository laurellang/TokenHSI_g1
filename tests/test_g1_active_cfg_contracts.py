from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml"


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
    files = [
        ROOT / "tools/validate_g1_interfaces.py",
        ROOT / "tokenhsi/scripts/single_task/g1_tokenhsi_traj_train.sh",
        ROOT / "TOKENHSI_G1_REMOTE_RUNBOOK.md",
    ]

    for path in files:
        text = path.read_text()
        assert cfg_ref in text
        assert stale_ref not in text
