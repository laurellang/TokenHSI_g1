# G1 Body / Asset Status

## Active Training Asset

Current training config uses:

```text
tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf
```

Configured in:

```text
tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml
```

The earlier file:

```text
assets/unitree_g1/g1_29dof_with_hand_rev_1_0.urdf
```

is not the active training asset unless the config is changed.

## GMR Retarget Asset

The GMR retarget XML originally referenced is:

```text
/home/lenovo/Projects/GMR/assets/unitree_g1/g1_mocap_29dof_with_hands.xml
```

The GMR motion schema and the active URDF both expose 43 movable joints, and the joint order was checked to match.

## Unitree Official Asset Check

Downloaded / checked source:

```text
https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description
```

Important finding:

- Unitree official `g1_29dof_with_hand_rev_1_0.urdf` matches local `assets/unitree_g1/g1_29dof_with_hand_rev_1_0.urdf`.
- It does not directly replace the active training path unless copied/configured.
- The active `g1_dex3_ori.urdf` has the same link count and DOF order, but differs in some joint limits / sensor pose details.

## Body Schema Gap

URDF has links that are not present in the GMR motion body schema, including examples such as:

```text
head_link
left_hand_palm_link
right_hand_palm_link
d435_link
mid360_link
imu_in_torso
imu_in_pelvis
logo_link
pelvis_contour_link
```

GMR also has motion-only bodies such as toe/fingertip bodies that are not one-to-one URDF links.

Fix already added:

- `tokenhsi/utils/body_schema.py`
- mapping of URDF-only links to nearest mapped motion ancestor plus URDF rest offset
- `head_link` termination-height lookup instead of hardcoded `head`

Related tests passed previously:

```text
tests/test_body_schema_mapping.py
tests/test_g1_active_cfg_contracts.py
```

## IsaacGym Smoke Checks

No-action / no-gravity IsaacGym smoke check showed:

```text
bodies: 53
dofs: 43
head_link - torso_link distance stayed constant
```

The later short-training policy rollout also showed `head_link` and `torso_link` distance nearly constant numerically:

```text
frames: 180
bodies: 53
head_torso_dist_delta: 3.8743019104003906e-07
```

This suggests the fixed joint is not physically breaking in IsaacGym.

## Current Problem

The Viser mesh visualization is still visually wrong:

- head still appears wrong / flying
- after applying inertial-origin correction, the joints / body arrangement look even more乱
- therefore the current Viser replay is not yet a trusted visual explanation of the IsaacGym state

Important distinction:

```text
Rigid-body distance checks say head_link is not physically separating.
Mesh replay visualization is still frame-wrong or convention-wrong.
```

So the current issue is likely one of:

1. IsaacGym rigid body state frame is not the same frame assumed by the Viser replay.
2. URDF link frame, inertial frame, collision frame, and visual mesh frame are being composed incorrectly.
3. IsaacGym may apply its own URDF import transforms differently from the simple formula used in `tools/visualize_g1_smoke_rollout.py`.
4. Saved policy rollout body poses may be correct, but mesh placement in Viser is wrong.

## Viser Tools Added

Current Viser replay script:

```text
tools/visualize_g1_smoke_rollout.py
```

It can:

- record no-action IsaacGym smoke body states
- replay saved `.npz` as URDF visual meshes
- replay short-training policy rollout `.npz`

Current warning:

```text
Do not use this mesh replay as authoritative yet.
The frame convention for mesh placement is still unresolved.
```

## Short Training Status

Short training smoke test was run on remote:

```text
/home/zxLang/TokenHSI_g1
```

Important environment note:

```bash
source /home/zxLang/miniconda3/etc/profile.d/conda.sh
conda activate tokenhsi_g1
```

Without activating the conda env, `gymtorch` failed because `ninja` was not on PATH.

Short training checkpoint:

```text
output/g1_dex3_short_train_20260717_174807/Humanoid_17-17-48-09/nn/Humanoid.pth
```

Policy rollout:

```text
/tmp/g1_short_train_policy_rollout.npz
```

## Next Debug Step

The next useful step is not more training. First resolve mesh/body frame convention.

Recommended next check:

1. Build a direct IsaacGym visual-shape transform dump if the API exposes rigid shape local poses.
2. Compare IsaacGym's actor rigid shape transforms against URDF visual/inertial origins.
3. In Viser, draw three frames per link separately:
   - IsaacGym rigid body pose
   - URDF inertial frame
   - URDF visual mesh frame
4. Start with only `torso_link` and `head_link`, not the full robot.

Until that is fixed, Viser mesh output can be misleading even if the underlying IsaacGym rigid bodies are stable.
