# G1 Viser Smoke Rollout Runbook

This note records the exact path that worked for visualizing the IsaacGym G1 smoke test as URDF meshes in Viser.

## What This Visualization Shows

- It replays a saved IsaacGym no-policy smoke trajectory.
- It uses the current training URDF visual meshes, not a point skeleton.
- It places each link mesh from saved rigid-body `body_pos` and `body_quat`.
- `head_link` is colored red, `torso_link` is colored blue, other links are gray.
- The viewer draws a `torso_link -> head_link` segment and shows the current distance.

For the verified no-action/no-gravity smoke test, the head does not separate from the torso:

```text
rollout /tmp/g1_smoke_rollout.npz frames 300 bodies 53
head_torso_dist_min 0.04417824372649193
head_torso_dist_max 0.04417824372649193
head_torso_dist_delta 0.0
```

## Files Added

```text
tools/visualize_g1_smoke_rollout.py
tests/test_visualize_g1_smoke_rollout.py
```

Local and remote test command:

```bash
/home/zxLang/miniconda3/envs/tokenhsi_g1/bin/python -m pytest tests/test_visualize_g1_smoke_rollout.py -q
```

Expected:

```text
4 passed
```

## Remote Workflow

Run on `zxLang@10.128.0.159` in `/home/zxLang/TokenHSI_g1`.

Set the library path before IsaacGym commands:

```bash
export LD_LIBRARY_PATH=/home/zxLang/miniconda3/envs/tokenhsi_g1/lib:${LD_LIBRARY_PATH}
```

Record the no-action/no-gravity smoke trajectory:

```bash
/home/zxLang/miniconda3/envs/tokenhsi_g1/bin/python tools/visualize_g1_smoke_rollout.py record \
  --asset-root tokenhsi/data/assets \
  --asset-file mjcf/g1_dex3_ori.urdf \
  --output /tmp/g1_smoke_rollout.npz \
  --steps 300 \
  --disable-gravity
```

Start the mesh replay viewer:

```bash
nohup /home/zxLang/miniconda3/envs/tokenhsi_g1/bin/python tools/visualize_g1_smoke_rollout.py view \
  --rollout /tmp/g1_smoke_rollout.npz \
  --urdf tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf \
  --host 0.0.0.0 \
  --port 8080 \
  > /tmp/g1_smoke_rollout_viser.log 2>&1 &
```

Forward to local:

```bash
ssh -N -L 18080:127.0.0.1:8080 zxLang@10.128.0.159
```

Open locally:

```text
http://127.0.0.1:18080
```

Verify local tunnel:

```bash
curl -s -o /tmp/viser_smoke_rollout_check.html -w "%{http_code} %{size_download}\n" http://127.0.0.1:18080
```

Expected:

```text
200 2888259
```

The byte count can change with Viser versions, but HTTP status must be `200`.

## Pitfalls From This Debug Session

Read this section before re-running the visualization.

1. Do not use `gymtorch` for this smoke recording path.

   The remote environment failed with:

   ```text
   RuntimeError: Ninja is required to load C++ extensions
   ```

   The working script uses IsaacGym's CPU structured API:

   ```python
   gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
   ```

2. Keep `LD_LIBRARY_PATH` set for IsaacGym.

   Without the conda env lib directory, IsaacGym can fail on `libpython3.8.so.1.0`.

3. Check port `8080` before starting Viser.

   Old Viser viewers can keep the port:

   ```bash
   ss -ltnp 'sport = :8080'
   ```

   If the process is an old viewer, stop only that PID.

4. The local URL is `18080`, not `8080`.

   Remote Viser listens on remote `8080`; local browser uses:

   ```text
   http://127.0.0.1:18080
   ```

5. The SSH tunnel can die independently of the remote viewer.

   If remote Viser is alive but local curl returns `000 0`, recreate:

   ```bash
   ssh -N -L 18080:127.0.0.1:8080 zxLang@10.128.0.159
   ```

6. This visualization uses the active training asset.

   The current training config loads:

   ```text
   tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf
   ```

   Do not accidentally visualize only:

   ```text
   assets/unitree_g1/g1_29dof_with_hand_rev_1_0.urdf
   ```

   unless the goal is asset comparison.

7. "rollout" is acceptable wording, but this specific case is more precisely a smoke-test trajectory.

   It has no policy action. It is no-action/no-gravity unless command-line flags are changed.

## Quick Health Check

After recording, verify head stability numerically:

```bash
/home/zxLang/miniconda3/envs/tokenhsi_g1/bin/python - <<'PY'
import numpy as np
p = "/tmp/g1_smoke_rollout.npz"
d = np.load(p)
names = [str(x) for x in d["body_names"].tolist()]
pos = d["body_pos"]
hi = names.index("head_link")
ti = names.index("torso_link")
dist = np.linalg.norm(pos[:, hi] - pos[:, ti], axis=-1)
print("rollout", p, "frames", pos.shape[0], "bodies", pos.shape[1])
print("head_torso_dist_min", float(dist.min()))
print("head_torso_dist_max", float(dist.max()))
print("head_torso_dist_delta", float(dist.max() - dist.min()))
PY
```

For the verified smoke test, `head_torso_dist_delta` was `0.0`.
