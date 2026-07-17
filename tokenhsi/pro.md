# Training Smoke Test Issue

Short training command attempted on remote:

```bash
cd /home/zxLang/TokenHSI_g1
source /home/zxLang/miniconda3/etc/profile.d/conda.sh
conda activate tokenhsi_g1
export PYTHONPATH=.:tokenhsi:${PYTHONPATH}
export LD_LIBRARY_PATH=/home/zxLang/miniconda3/envs/tokenhsi_g1/lib:${LD_LIBRARY_PATH}

/home/zxLang/miniconda3/envs/tokenhsi_g1/bin/python ./tokenhsi/run.py \
  --task HumanoidTrajSitCarryClimb \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
  --cfg_env tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml \
  --motion_file tokenhsi/data/dataset_g1_all.yaml \
  --num_envs 64 \
  --horizon_length 64 \
  --minibatch_size 4096 \
  --max_iterations 1 \
  --headless \
  --output_path output/g1_dex3_short_train_20260717_174434
```

Failure:

```text
RuntimeError: Ninja is required to load C++ extensions
```

Cause:

- `tokenhsi/run.py` imports IsaacGym `gymtorch` through `env.tasks.vec_task`.
- `gymtorch` compiles/loads a PyTorch C++ extension.
- The first attempt used the env Python directly instead of activating `tokenhsi_g1`, so the shell PATH did not include the conda env binaries.

Next check:

```bash
source /home/zxLang/miniconda3/etc/profile.d/conda.sh
conda activate tokenhsi_g1
export PATH=/home/zxLang/miniconda3/envs/tokenhsi_g1/bin:${PATH}
which ninja
```

If `ninja` is still missing, install a working ninja binary into `tokenhsi_g1` or system PATH before rerunning short training.

## Resolved During Short Training Smoke Test

The fix was to activate the environment before running training:

```bash
source /home/zxLang/miniconda3/etc/profile.d/conda.sh
conda activate tokenhsi_g1
```

After activation, `which ninja` resolved to:

```text
/home/zxLang/miniconda3/envs/tokenhsi_g1/bin/ninja
```

The short training smoke test then passed and saved:

```text
output/g1_dex3_short_train_20260717_174807/Humanoid_17-17-48-09/nn/Humanoid.pth
```

A policy rollout was saved from that checkpoint:

```text
/tmp/g1_short_train_policy_rollout.npz
frames: 180
bodies: 53
head_torso_dist_delta: 3.8743019104003906e-07
root_z_min: 0.6076019406318665
root_z_max: 0.8460065126419067
```

It is being replayed by Viser on remote port `8080`, forwarded locally to:

```text
http://127.0.0.1:18080
```

