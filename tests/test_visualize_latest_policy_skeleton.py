import argparse
import os
import sys
from pathlib import Path

from tools import visualize_latest_policy_skeleton as viz


def test_find_latest_checkpoint_uses_newest_pth_mtime(tmp_path: Path):
    older = tmp_path / "run_a" / "Humanoid_1" / "nn" / "Humanoid.pth"
    newer = tmp_path / "run_b" / "Humanoid_2" / "nn" / "Humanoid_0001000.pth"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (100.0, 100.0))
    os.utime(newer, (200.0, 200.0))

    assert viz.find_latest_checkpoint(tmp_path) == newer


def test_build_record_command_sets_checkpoint_and_rollout_env(tmp_path: Path):
    checkpoint = tmp_path / "output" / "run" / "Humanoid" / "nn" / "Humanoid.pth"
    rollout = tmp_path / "policy_rollout.npz"
    args = argparse.Namespace(
        python=sys.executable,
        task="HumanoidTrajSitCarryClimb",
        cfg_train="train.yaml",
        cfg_env="env.yaml",
        motion_file="motions.yaml",
        num_envs=1,
        output_path="output/latest_policy_rollout_eval",
        rl_device="cuda:0",
        sim_device="cuda:0",
        pipeline="gpu",
        graphics_device_id=0,
        steps=240,
        env_id=0,
    )

    cmd, env = viz.build_record_command(args, checkpoint, rollout)

    assert cmd[:2] == [sys.executable, "./tokenhsi/run.py"]
    assert "--play" in cmd
    assert cmd[cmd.index("--checkpoint") + 1] == str(checkpoint)
    assert cmd[cmd.index("--num_envs") + 1] == "1"
    assert env["TOKENHSI_RECORD_ROLLOUT_NPZ"] == str(rollout)
    assert env["TOKENHSI_RECORD_ROLLOUT_MAX_FRAMES"] == "240"
