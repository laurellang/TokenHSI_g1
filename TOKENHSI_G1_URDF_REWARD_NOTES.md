# TokenHSI G1 改动对照

本文只记录两类内容：

1. `g1_dex3.urdf`：改前是什么，改后是什么
2. reward / task cfg：改前是什么，改后是什么

## 1. URDF 改动对照

改前参考：

```text
/home/lenovo/Projects/TokenHSI/assets/unitree_g1_description/g1_29dof_with_hand_rev_1_0.urdf
```

改后文件：

```text
tokenhsi/data/assets/mjcf/g1_dex3.urdf
```

| 项目 | 改前 | 改后 | 目的 |
| --- | --- | --- | --- |
| 文件位置 | 原始 Unitree G1 + Dex3 URDF 在 `assets/unitree_g1_description/`。 | 训练使用的 URDF 放到 `tokenhsi/data/assets/mjcf/g1_dex3.urdf`。 | 让 TokenHSI cfg 可以通过 `assetRoot: tokenhsi/data/assets` 加载。 |
| active asset 指向 | `assetFileName` 不是这个 URDF，原 TokenHSI 用 `mjcf/phys_humanoid_v3.xml`，早期 G1 版本用过 MJCF。 | `assetFileName: "mjcf/g1_dex3.urdf"`。 | 训练直接加载 GMR retarget 对应的 G1 + Dex3 机器人。 |
| link 数 | `53` 个 links。 | `52` 个 links。 | 对齐 GMR retarget motion pkl 里的 `link_body_list`。 |
| joint 数 | `52` 个 joints，其中 `43` 个 movable joints。 | `51` 个 joints，其中 `43` 个 movable joints。 | 保持 policy/action DOF 为 43，同时整理 fixed link/joint。 |
| movable DOF | `43`。 | `43`。 | 不改变 G1 + Dex3 的可控关节数。 |
| link 顺序 | 原始 URDF 顺序包含 contour/logo/sensor/palm 等 fixed links，和 retarget `link_body_list` 不一致。 | link 顺序按 retarget motion schema 整理。 | AMP obs、key body、motion state 都依赖 body 顺序。 |
| 删除的 fixed links | 包含 `pelvis_contour_link`, `logo_link`, `head_link`, `imu_in_torso`, `imu_in_pelvis`, `d435_link`, `mid360_link`, `left_hand_palm_link`, `right_hand_palm_link`。 | 这些 fixed links 已移除。 | 去掉 retarget schema 中没有的 body，避免 body 数和顺序不一致。 |
| 删除的 fixed joints | 包含 `pelvis_contour_joint`, `logo_joint`, `head_joint`, `imu_in_torso_joint`, `imu_in_pelvis_joint`, `d435_joint`, `mid360_joint`, `left_hand_palm_joint`, `right_hand_palm_joint`。 | 这些 fixed joints 已移除。 | 与删除 fixed links 对应。 |
| 新增 toe links | 原始 URDF 没有 `left_toe_link`, `right_toe_link`。 | 新增 `left_toe_link`, `right_toe_link` 及 fixed joints。 | 补齐 retarget `link_body_list` 所需 body。 |
| 新增 fingertip links | 原始 URDF 没有 thumb/middle/index fingertip links。 | 新增左右手 `thumb/middle/index_finger_tip` links 及 fixed joints。 | 补齐 retarget `link_body_list` 所需手指末端 body。 |
| 手掌父子关系 | 手指 base joints 的 parent 是 `left_hand_palm_link` / `right_hand_palm_link`。 | palm links 被移除，手指 base joints 改挂到 `left_wrist_yaw_link` / `right_wrist_yaw_link`，并调整 origin 保持全局位置。 | 删除 palm fixed link 后仍保持手指位置不漂。 |
| 左手 thumb base | parent: `left_hand_palm_link`, origin: `0.0255 0 0`。 | parent: `left_wrist_yaw_link`, origin: `0.067 0.003 0`。 | reparent 后保持位置一致。 |
| 左手 middle/index base | parent: `left_hand_palm_link`, origin: `0.0777 0.0016 +/-0.0285`。 | parent: `left_wrist_yaw_link`, origin: `0.1192 0.0046 +/-0.0285`。 | reparent 后保持位置一致。 |
| 右手 thumb base | parent: `right_hand_palm_link`, origin: `0.0255 0 0`。 | parent: `right_wrist_yaw_link`, origin: `0.067 -0.003 0`。 | reparent 后保持位置一致。 |
| 右手 middle/index base | parent: `right_hand_palm_link`, origin: `0.0777 -0.0016 +/-0.0285`。 | parent: `right_wrist_yaw_link`, origin: `0.1192 -0.0046 +/-0.0285`。 | reparent 后保持位置一致。 |
| mesh 路径 | `meshes/*.STL`。 | `../meshes/unitree_g1/*.STL`。 | 适配当前 `g1_dex3.urdf` 放在 `tokenhsi/data/assets/mjcf/` 下的位置。 |
| collision mesh | `38` 个 mesh collision。 | `0` 个 mesh collision。 | 避免 IsaacGym 加载 mesh collision / VHACD 时 native crash。 |
| cylinder collision | `4` 个 cylinder collision。 | `0` 个 cylinder collision。 | 避免 IsaacGym cylinder tessellation 路径。 |
| box collision | `2` 个 box collision。 | `33` 个 box collision。 | 用 primitive collision 近似机器人碰撞体。 |
| sphere collision | `8` 个 sphere collision。 | `9` 个 sphere collision。 | 用 primitive collision 近似脚部等接触点。 |
| runtime cylinder 替换 | 曾考虑让 IsaacGym runtime 把 cylinder 替换成 capsule。 | `replace_cylinder_with_capsule = False`，URDF 中也不保留 cylinder collision。 | 不让 IsaacGym 静默改变碰撞构型。 |
| actuator effort | 原 MJCF 可以从 actuator props 取 motor effort；URDF 加载时 actuator props 为空。 | 如果 actuator props 为空，就从 URDF DOF properties 的 `effort` 读取 43 个 effort。 | 修复 `get_asset_actuator_properties count=0` 导致的空列表问题。 |

## 2. Reward / Task CFG 改动对照

改前参考：

```text
/home/lenovo/Projects/TokenHSI/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml
```

改后文件：

```text
tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml
```

| 项目 | 改前 | 改后 | 目的 |
| --- | --- | --- | --- |
| 机器人资产 | `mjcf/phys_humanoid_v3.xml`。 | `mjcf/g1_dex3.urdf`。 | 从 humanoid_v3 训练改为 G1 + Dex3 训练。 |
| motion file | 原 cfg 不直接指定 G1 all motion。 | `motion_file: "tokenhsi/data/dataset_g1_all.yaml"`。 | 使用 GMR retarget 到 G1 的 motion 数据。 |
| key bodies | `["right_hand", "left_hand", "right_foot", "left_foot"]`。 | `["right_wrist_yaw_link", "left_wrist_yaw_link", "right_ankle_roll_link", "left_ankle_roll_link"]`。 | 换成 G1 URDF 中真实存在、且和 retarget schema 对齐的 body 名。 |
| contact bodies | `["right_foot", "left_foot"]`。 | `["right_ankle_roll_link", "left_ankle_roll_link"]`。 | G1 没有 humanoid_v3 的 foot body 名，需要改成 ankle/foot 接触 body。 |
| terminationHeight | `0.15`。 | `0.15`，暂未改值。 | 先保持原值；如果 G1 episode 长期过短，再优先调这个。 |
| task 采样 | `[0.1, 0.3, 0.3, 0.3]`，四类任务混训。 | `[1.0, 0.0, 0.0, 0.0]`，只采样 `traj/loco`。 | 当前阶段先只训练 loco。 |
| discriminator demo 采样 | loco/sit/carry/climb 多 skill 混合采样。 | `skillDiscProb: [1.0, 0.0, ...]`，只采 loco demo。 | AMP discriminator 当前只学习 loco motion prior。 |
| inactive task assets | 原 cfg 没有 `loadInactiveTaskAssets`，任务类会预加载 sit/climb object assets。 | `loadInactiveTaskAssets: False`，inactive sit/climb 使用 primitive proxy。 | 只训 loco 时避免加载 mesh-heavy chair/table/cabinet，降低 IsaacGym crash 风险。 |
| task mask 维度 | 四任务 mask/one-hot。 | 仍保留四任务 mask/one-hot。 | 严格保留 TokenHSI multi-task transformer 接口；只训 loco 也不改成 1 维。 |
| traj speed | `speedMin: 0.5`, `speedMax: 1.5`。 | 不变。 | loco 目标速度先沿用原设置。 |
| sit tarSpeed | 原 cfg 未显式设置，代码默认 `1.5`。 | 显式设置 `tarSpeed: 1.0`。 | G1 交互任务接近速度先调保守。 |
| climb object categories | `Box`, `Cabinet`, `Table_Square`。 | 先只保留 `Box`。 | 避免 table/cabinet mesh-heavy asset 在 smoke/loco 阶段触发加载问题。 |
| climb tarSpeed | 原 cfg未显式设置，代码默认 `1.5`。 | 显式设置 `tarSpeed: 1.0`。 | 降低 G1 爬台/接近物体速度要求。 |
| climb feet height reward scale | 原 cfg 未显式设置，代码默认值生效。 | 显式设置 `feetHeightRewardScale: 30.0`。 | 把 G1 climb 相关高度 reward 参数放进 cfg，便于后续调参。 |
| carry tarSpeed | 原 cfg 未显式设置，代码默认 `1.5`。 | 显式设置 `tarSpeed: 1.0`。 | 降低 G1 搬运目标速度要求。 |
| carry hand reward active distance | 原 cfg 未显式设置，代码默认 `0.7`。 | 显式设置 `handRewardActiveDist: 0.7`。 | 当前值不变，但放进 cfg 便于后续调参。 |
| carry box lift height margin | 原 cfg 未显式设置，代码默认 `0.2`。 | 显式设置 `boxLiftHeightMargin: 0.15`。 | 调小抬箱判定 margin，使其更适合 G1 和缩小后的箱子尺寸。 |
| box base size | `[0.4, 0.4, 0.4]`。 | `[0.337447, 0.337447, 0.337447]`。 | 按 G1/humanoid_v3 身高比例缩放：`0.4 * 0.793 / 0.94 = 0.337447`。 |
| box test sizes | `[0.22 ... 0.57]` 等 humanoid_v3 尺寸。 | 全部乘以 `0.793 / 0.94`，例如 `0.22 -> 0.185596`，`0.57 -> 0.480862`。 | 保持原尺寸分布形状，但绝对尺寸适配 G1。 |
| box random scale range | `[0.5, 1.5]`。 | 不变。 | 因为 base size 已经缩小，随机比例先保持原设置。 |
| power reward | `power_reward: True`, `power_coefficient: 0.0005`。 | 不变。 | 能耗惩罚入口先不动。 |
| task reward 公式 | 使用原始 `compute_traj_reward / compute_sit_reward / compute_climb_reward / compute_carry_reward`。 | 公式未重写。 | 当前阶段只做 G1 参数和接口适配，不大改 TokenHSI reward 框架。 |
| AMP discriminator reward | 原 AMP reward 公式和 `task_reward_w / disc_reward_w` 混合。 | 暂未改公式和权重。 | 先保证 sim AMP obs 和 demo AMP obs 的 G1 schema 对齐；后续再视训练效果调 AMP/task reward balance。 |

当前只训 loco 时，真正参与 task reward 的主要是：

```text
compute_traj_reward(root_pos, tar_pos)
```

但 obs / mask / network 仍保持四任务接口：

```text
[self_obs 778] + [traj 20 | sit 38 | carry 42 | climb 27] + [task one-hot 4] = 909
```
