# G1 Reward Terms

当前 G1 multi-task 环境的四个基础 task 是 `traj`、`sit`、`carry`、`climb`。其中 `traj` 对应基础 loco 任务。

Reward 运行时代码集中在：

- `tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py`

Reward 开关、系数和 task/skill 配置集中在：

- `tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml`

| 基础 task | 代码 task 名 | 对应 skills | reward 入口 | reward term 名称 | 配置项 |
|---|---|---|---|---|---|
| loco | `traj` | `loco` | `compute_g1_loco_reward()` | `pos_reward`, `track_lin_vel_reward`, `track_ang_vel_reward`, `alive`, `lin_vel_z_penalty`, `ang_vel_xy_penalty`, `flat_orientation_penalty`, `base_height_penalty`, `joint_vel_penalty`, `action_rate_penalty`, `dof_pos_limits_penalty`, `feet_slide_penalty`, `feet_clearance_reward` | `traj.reward.*`, `traj.speedMin`, `traj.speedMax`, `traj.accelMax`, `traj.failDist` |
| sit | `sit` | `loco_sit`, `sit` | `compute_sit_reward()` | `reward_far_pos`, `reward_far_vel`, `reward_far_final`, `reward_near`, `root_vel_penalty`, `root_z_ang_vel_penalty` | `sit.tarSpeed`, `sit.sit_vel_penalty`, `sit.sit_vel_pen_coeff`, `sit.sit_vel_pen_threshold`, `sit.sit_ang_vel_pen_coeff`, `sit.sit_ang_vel_pen_threshold` |
| carry | `carry` | `loco_carry`, `pickUp`, `carryWith`, `putDown`, `omomo` | `compute_walk_reward()` + `compute_carry_reward()` + `compute_handheld_reward()` + `compute_putdown_reward()` | `walk_pos_reward`, `walk_vel_reward`, `carry_pos_reward_far`, `carry_pos_reward_near`, `carry_vel_reward`, `hands2box`, `putdown_reward`, `box_vel_penalty` | `carry.onlyVelReward`, `carry.onlyHeightHandHeldReward`, `carry.box_vel_penalty`, `carry.box_vel_pen_coeff`, `carry.box_vel_pen_threshold`, `carry.tarSpeed`, `carry.handRewardActiveDist`, `carry.boxLiftHeightMargin` |
| climb | `climb` | `loco_climb`, `climb`, `climbNoRSI` | `compute_climb_reward()` | `pos_reward`, `vel_reward`, `pos_reward_near`, `feet_height_reward`, `root_vel_penalty` | `climb.tarSpeed`, `climb.feetHeightRewardScale`, `climb.climb_vel_penalty`, `climb.climb_vel_pen_coeff`, `climb.climb_vel_pen_threshold` |
| all tasks | all | all | `_compute_reward()` | `power_reward` | `power_reward`, `power_coefficient` |

## Runtime Composition

| task | reward composition |
|---|---|
| `traj` | `reward = posWeight * pos_reward + trackLinVelWeight * track_lin_vel_reward + trackAngVelWeight * track_ang_vel_reward + aliveWeight + weighted stability/smoothness/feet terms` |
| `sit` | `reward = 0.7 * reward_near + 0.3 * reward_far_final + optional root_vel_penalty + optional root_z_ang_vel_penalty` |
| `carry` | `reward = walk_r + carry_r + handheld_r + putdown_r` |
| `climb` | `reward = 0.2 * vel_reward + 0.5 * pos_reward_near + 0.3 * feet_height_reward + optional root_vel_penalty` |
| all | if `power_reward=True`, final `rew_buf = task_reward + power_reward`, where `power_reward = -power_coefficient * sum(abs(dof_force * dof_vel))` |
