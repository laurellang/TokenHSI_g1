#!/usr/bin/env python3
"""Compare G1 AMP demo observations against reset/sim observation paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
TOKENHSI = ROOT / "tokenhsi"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOKENHSI))

from isaacgym import gymapi, gymtorch  # noqa: E402
import torch  # noqa: E402
from utils.config import load_cfg, parse_sim_params, set_np_formatting, set_seed  # noqa: E402
from utils.parse_task import parse_task  # noqa: E402
from tokenhsi.env.tasks.multi_task.humanoid_traj_sit_carry_climb import build_amp_observations  # noqa: E402


def make_args(num_envs: int, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        cfg_train="tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml",
        cfg_env="tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml",
        cfg_task_plan="",
        task="HumanoidTrajSitCarryClimb",
        task_type="Python",
        headless=True,
        randomize=False,
        torch_deterministic=False,
        num_envs=num_envs,
        episode_length=0,
        seed=42,
        max_iterations=0,
        horizon_length=-1,
        minibatch_size=-1,
        output_path="output/amp_obs_alignment_diag",
        logdir="logs/",
        experiment="Base",
        metadata=False,
        checkpoint="Base",
        resume=0,
        llc_checkpoint="",
        llc_checkpoint_2="",
        llc_checkpoint_3="",
        llc_checkpoint_4="",
        llc_checkpoint_5="",
        llc_checkpoint_6="",
        llc_checkpoint_7="",
        hrl_checkpoint="",
        eval=False,
        eval_task="",
        test=False,
        play=False,
        train=True,
        record=False,
        render_device="",
        horovod=False,
        rl_device=device,
        device=device,
        device_id=0,
        compute_device_id=0,
        graphics_device_id=0,
        sim_device_type=device,
        use_gpu=False,
        use_gpu_pipeline=False,
        physics_engine=gymapi.SIM_PHYSX,
        slices=0,
        subscenes=0,
        num_threads=4,
    )


def segment_slices(task):
    base = 0
    parts = []
    if task._root_height_obs:
        parts.append(("root_h", slice(base, base + 1)))
    else:
        parts.append(("root_h_zero", slice(base, base + 1)))
    base += 1
    parts.extend(
        [
            ("root_rot_6d", slice(base, base + 6)),
            ("root_vel", slice(base + 6, base + 9)),
            ("root_ang_vel", slice(base + 9, base + 12)),
            ("dof_obs", slice(base + 12, base + 12 + task._dof_obs_size)),
            (
                "dof_vel",
                slice(base + 12 + task._dof_obs_size, base + 12 + task._dof_obs_size + task.num_dof),
            ),
            (
                "key_body_pos",
                slice(
                    base + 12 + task._dof_obs_size + task.num_dof,
                    base + 12 + task._dof_obs_size + task.num_dof + 3 * len(task._key_body_ids),
                ),
            ),
        ]
    )
    if task._enable_task_specific_disc:
        parts.append(("task_mask", slice(task._num_amp_obs_per_step - task._num_tasks, task._num_amp_obs_per_step)))
    return parts


def summarize_delta(name: str, actual: torch.Tensor, expected: torch.Tensor):
    delta = (actual - expected).detach()
    print(
        f"{name:14s} max_abs={delta.abs().max().item():.6g} "
        f"mean_abs={delta.abs().mean().item():.6g} rms={torch.sqrt((delta * delta).mean()).item():.6g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skill", default="loco")
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--time", type=float, default=0.5)
    args_cli = parser.parse_args()

    set_np_formatting()
    args = make_args(args_cli.num_envs, args_cli.device)
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    try:
        cfg, cfg_train, _ = load_cfg(args)
    finally:
        sys.argv = old_argv

    cfg_train["params"]["seed"] = cfg_train["params"]["config"]["seed"] = set_seed(42, False)
    cfg["env"]["numEnvs"] = args_cli.num_envs
    cfg["env"]["stateInit"] = "Start"
    cfg["env"]["taskInitProb"] = [1.0, 0.0, 0.0, 0.0]
    cfg["env"]["skillDiscProb"] = [1.0] + [0.0] * (len(cfg["env"]["skillDiscProb"]) - 1)
    cfg["env"]["loadInactiveTaskAssets"] = False

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, _ = parse_task(args, cfg, cfg_train, sim_params)

    motion_lib = task._motion_lib[args_cli.skill]
    env_ids = torch.arange(args_cli.num_envs, device=task.device, dtype=torch.long)
    motion_ids = torch.full((args_cli.num_envs,), args_cli.motion_id, device=task.device, dtype=torch.long)
    motion_times = torch.full((args_cli.num_envs,), args_cli.time, device=task.device, dtype=torch.float32)

    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = motion_lib.get_motion_state(
        motion_ids, motion_times
    )
    demo = build_amp_observations(
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        key_pos,
        task._local_root_obs,
        task._root_height_obs,
        task._dof_obs_size,
        task._dof_offsets,
    )
    if task._enable_task_specific_disc:
        mask = torch.zeros((args_cli.num_envs, task._num_tasks), device=task.device)
        mask[:, task.TaskUID["traj"].value] = 1.0
        demo = torch.cat([demo, mask], dim=-1)

    task._set_env_state(env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel)
    body_pos, body_rot, body_vel, body_ang_vel = motion_lib.get_motion_state_max(motion_ids, motion_times)
    body_state = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)
    task._kinematic_humanoid_rigid_body_states[env_ids] = task._map_motion_body_state_to_asset(body_state)
    task._task_mask[env_ids] = 0
    task._task_mask[env_ids, task.TaskUID["traj"].value] = 1
    task._compute_amp_observations(env_ids)
    reset_path = task._curr_amp_obs_buf[env_ids].clone()

    print("=== pure reset-path vs direct demo builder ===")
    print(f"num_dof={task.num_dof} per_step={task._num_amp_obs_per_step} amp_steps={task._num_amp_obs_steps}")
    for name, slc in segment_slices(task):
        summarize_delta(name, reset_path[:, slc], demo[:, slc])

    print("=== isaac sim refreshed path vs direct demo builder ===")
    actor_ids = task._humanoid_actor_ids[env_ids].to(torch.int32)
    task.gym.set_actor_root_state_tensor_indexed(
        task.sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
    )
    task.gym.set_dof_state_tensor_indexed(
        task.sim, gymtorch.unwrap_tensor(task._dof_state), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
    )
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task.gym.refresh_actor_root_state_tensor(task.sim)
    task.gym.refresh_dof_state_tensor(task.sim)
    task.gym.refresh_rigid_body_state_tensor(task.sim)
    task._compute_amp_observations()
    sim_path = task._curr_amp_obs_buf[env_ids].clone()
    for name, slc in segment_slices(task):
        summarize_delta(name, sim_path[:, slc], demo[:, slc])

    task.gym.destroy_sim(task.sim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
