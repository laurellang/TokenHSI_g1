# TokenHSI_g1

这是 TokenHSI 的 G1 + Dex3 训练版本。当前目标是把原 TokenHSI 的 unified transformer + AMP 框架迁移到 G1 机器人上，用 GMR retarget 后的 G1 motion 数据训练 `traj / sit / carry / climb` 四类 foundational HSI task。

原版 TokenHSI 论文与代码入口：

- Paper: https://arxiv.org/abs/2503.19901
- Project: https://liangpan99.github.io/TokenHSI/
- Upstream code: https://github.com/liangpan99/TokenHSI

## 当前版本

核心文件：

```text
TokenHSI_g1
|-- tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml
|-- tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml
|-- tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf
|-- tokenhsi/data/dataset_g1_all.yaml
|-- tokenhsi/data/dataset_sit/objects
|-- tokenhsi/data/dataset_amass_climb/objects
|-- tokenhsi/scripts/single_task/g1_tokenhsi_traj_train.sh
|-- tools/validate_g1_interfaces.py
```

当前接口规格：

| Item | Value |
| --- | --- |
| Robot | G1 + Dex3 |
| Motion format | GMR retargeted robot motion |
| Action / DOF | `43` |
| Robot asset bodies | `53` |
| Dataset bodies | `52` |
| Tasks | `traj`, `sit`, `carry`, `climb` |
| Task one-hot | `4` |
| Task obs sizes | `[20, 38, 42, 27]` |
| Policy obs | `924` |
| AMP obs per step | `330` |
| AMP history steps | `10` |
| AMP obs total | `3300` |

Box/carry 相关尺寸已经按 G1 和 humanoid_v3 的高度比例缩放：

```text
ratio = 0.793 / 0.94 = 0.8436
base box size: 0.4m -> 0.337447m
train random side range: 0.337447 * [0.5, 1.5] = [0.1687m, 0.5062m]
```

## 机器选择

建议在 IsaacGym 兼容性稳定的服务器上训练，例如 RTX 3090、A10、A100、RTX 4090、H100。

不建议在 RTX 50 系列本地机器上直接跑 IsaacGym。原因是 IsaacGym Preview 4 较老，`gymtorch` CUDA extension 对 50 系列的新 GPU 架构支持不稳定，容易在编译或运行时遇到 `unsupported gpu architecture`、`no kernel image is available` 或 gymtorch build failure。

本地 RTX 50 系列机器可以做：

```text
离线接口验证
motion schema 检查
asset 路径检查
transformer CPU forward 检查
```

正式 IsaacGym smoke test 和训练建议放到服务器。

## 环境配置

推荐环境名固定为 `tokenhsi_g1`。

```bash
conda create -n tokenhsi_g1 python=3.8 -y
conda activate tokenhsi_g1
pip install --upgrade pip
```

安装 PyTorch。CUDA 11.8 示例：

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

安装 IsaacGym Preview 4：

```bash
cd /path/to/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .
```

安装项目依赖：

```bash
cd /path/to/TokenHSI_g1
pip install -r requirements.txt
pip install ninja
```

如果 `rl_games` 版本冲突，优先保留项目使用的 `rl_games==1.1.4`。

8 卡分布式训练还需要 Horovod。建议只在远程训练服务器安装：

```bash
HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL \
  pip install -r requirements-distributed.txt
```

## 上传到服务器

首次上传：

```bash
REMOTE_USER=你的用户名
REMOTE_HOST=服务器IP或域名
REMOTE_DIR=/服务器上的目标路径

scp -r /home/lenovo/Projects/TokenHSI_g1 ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/
```

非 22 端口：

```bash
scp -P 2222 -r /home/lenovo/Projects/TokenHSI_g1 user@server_ip:/home/user/projects/
```

后续同步推荐 `rsync`：

```bash
rsync -avh --progress /home/lenovo/Projects/TokenHSI_g1/ \
  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/TokenHSI_g1/
```

带端口：

```bash
rsync -avh --progress -e "ssh -p 2222" /home/lenovo/Projects/TokenHSI_g1/ \
  user@server_ip:/home/user/projects/TokenHSI_g1/
```

## 开始训练前的检查

进入项目：

```bash
cd /path/to/TokenHSI_g1
conda activate tokenhsi_g1
export PYTHONPATH=.:tokenhsi
```

先跑离线接口验证。这个脚本不启动 IsaacGym sim：

```bash
python tools/validate_g1_interfaces.py
```

成功时应看到类似输出：

```text
G1 interface validation passed
asset: mjcf/g1_dex3_ori.urdf
motion yaml: tokenhsi/data/dataset_g1_all.yaml
motion entry references checked: 139
key body ids: [41, 24, 13, 6]
dimensions: self_obs=793 task_obs=131 obs=924 actions=43 amp_step=330 amp_obs=3300
```

如果服务器还没装好 `rl_games`，可以先跳过网络 forward：

```bash
python tools/validate_g1_interfaces.py --skip-network
```

## IsaacGym smoke test

离线验证通过后，先用小规模环境跑 1 个 iteration：

```bash
cd /path/to/TokenHSI_g1
conda activate tokenhsi_g1
export PYTHONPATH=.:tokenhsi

python ./tokenhsi/run.py \
  --task HumanoidTrajSitCarryClimb \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
  --cfg_env tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml \
  --motion_file tokenhsi/data/dataset_g1_all.yaml \
  --num_envs 64 \
  --horizon_length 64 \
  --minibatch_size 4096 \
  --max_iterations 1 \
  --headless \
  --output_path output/g1_dex3_smoke
```

重点看这几行：

```text
num_actions: 43
num_obs: 909
```

理想情况是完成 1 个 iteration。至少也应该能进入 PPO/AMP step，而不是在 robot asset、motion、obs/action shape、network shape 阶段报错。

## 正式训练

smoke test 通过后，如果只跑单卡，运行训练脚本：

```bash
cd /path/to/TokenHSI_g1
conda activate tokenhsi_g1
export PYTHONPATH=.:tokenhsi

bash tokenhsi/scripts/single_task/g1_tokenhsi_traj_train.sh
```

脚本内容等价于：

```bash
python ./tokenhsi/run.py --task HumanoidTrajSitCarryClimb \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
  --cfg_env tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml \
  --motion_file tokenhsi/data/dataset_g1_all.yaml \
  --num_envs 4096 \
  --headless
```

如果显存不够，先把 `--num_envs 4096` 改成 `2048` 或 `1024`。如果仍然 OOM，再同步降低 `horizon_length` 或 `minibatch_size` 做短跑测试。

## 8 卡分布式训练

当前推荐架构是单机 8 卡 Horovod：每张 GPU 一个进程，每个进程本地创建 IsaacGym env，模型梯度通过 Horovod 同步，rank 0 负责日志和 checkpoint。

先用小规模 8 卡 smoke test：

```bash
cd /path/to/TokenHSI_g1
conda activate tokenhsi_g1
export PYTHONPATH=.:tokenhsi

NUM_GPUS=8 NUM_ENVS_PER_GPU=64 HORIZON_LENGTH=64 MINIBATCH_SIZE=4096 MAX_ITERATIONS=1 \
  bash tokenhsi/scripts/multi_gpu/g1_tokenhsi_8gpu_train.sh
```

正式 8 卡训练默认每卡 `4096` 个 env，总 env 数是 `32768`：

```bash
cd /path/to/TokenHSI_g1
conda activate tokenhsi_g1
export PYTHONPATH=.:tokenhsi

bash tokenhsi/scripts/multi_gpu/g1_tokenhsi_8gpu_train.sh
```

脚本中的 `--num_envs` 表示每张 GPU 的 env 数，不是全局 env 数。可以用环境变量覆盖：

```bash
NUM_GPUS=8 NUM_ENVS_PER_GPU=2048 bash tokenhsi/scripts/multi_gpu/g1_tokenhsi_8gpu_train.sh
```

## 训练输出

默认输出目录是：

```text
output/
```

可以用 TensorBoard 看训练曲线：

```bash
tensorboard --logdir output --host 0.0.0.0 --port 6006
```

如果在远程服务器上看：

```bash
ssh -L 6006:localhost:6006 user@server_ip
```

然后在本地浏览器打开：

```text
http://localhost:6006
```

## 常见失败分类

| 现象 | 优先检查 |
| --- | --- |
| URDF / mesh load error | `tokenhsi/data/assets/mjcf/g1_dex3_ori.urdf` 和 mesh 路径 |
| motion pkl 缺字段或 shape 不对 | `python tools/validate_g1_interfaces.py` |
| `StraightChair_Normal` / `Box` object path missing | 确认 `dataset_sit/objects` 和 `dataset_amass_climb/objects` 已一起上传 |
| `num_actions` 不是 `43` | URDF movable joint / DOF 接口 |
| `num_obs` 不是 `924` | self obs、task obs、one-hot 接口 |
| gymtorch build fail | PyTorch / CUDA / IsaacGym / GPU 架构 |
| 能跑但容易摔 | reward、termination、PD gain、contact、box/object 尺寸 |
| carry 抓箱不稳定 | box size、box lift margin、hand reward active distance |
| climb 脚抬不够 | climb feet height reward scale、object 高度、tar speed |

当前 `traj-only` smoke/训练配置里，`loadInactiveTaskAssets: False`，并且 `climb.objCategories` 先只保留 `Box`。原因是 IsaacGym 在预加载 mesh-heavy 的 chair / cabinet / table object 并做 mesh loading 或 VHACD convex decomposition 时可能 native segfault；虽然当前只训 traj，原任务类默认仍会预加载 sit/carry/climb assets。现在 inactive sit/climb 会用 tiny primitive box proxy 占位，不再加载它们的 mesh URDF。等进入 full sit/climb 训练时，再单独处理这些 mesh asset。

当前 active robot asset 是 `g1_dex3_ori.urdf`，保留原生 G1 + Dex3 结构。由于 GMR retarget motion schema 是 52 bodies，而原生 URDF 是 53 bodies，训练入口会按 body name 把 motion state 映射到当前 asset；缺失的 fixed/auxiliary link 使用最近可映射父 link 的 motion state 作为 kinematic reference。

## 当前调参入口

主要改这个文件：

```text
tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml
```

常用字段：

```yaml
env:
  terminationHeight: 0.15
  traj:
    speedMin: 0.5
    speedMax: 1.5
  sit:
    tarSpeed: 1.0
  climb:
    tarSpeed: 1.0
    feetHeightRewardScale: 30.0
  carry:
    tarSpeed: 1.0
    handRewardActiveDist: 0.7
    boxLiftHeightMargin: 0.15
    box:
      build:
        baseSize: [0.337447, 0.337447, 0.337447]
        scaleRangeX: [0.5, 1.5]
```

`discriminator` 的 AMP reward 先不建议改。现在更重要的是保证 sim AMP obs 和 demo AMP obs 的 G1 schema 对齐；如果后续训练出现“动作像但任务失败”或“任务完成但动作很怪”，再调 AMP reward scale 和 task reward balance。

## 本地可做的验证

本地没有 IsaacGym 或 GPU 架构不兼容时，仍然可以跑：

```bash
cd /home/lenovo/Projects/TokenHSI_g1
conda activate tokenhsi_g1
export PYTHONPATH=.:tokenhsi

python tools/validate_g1_interfaces.py --skip-network
python -m pytest -q tests
```

当前测试覆盖：

```text
GMRRobotMotionLib 插值
G1 cfg box 尺寸缩放
sit/climb object asset 目录结构
训练/验证入口 cfg 引用
URDF 可视化工具的基础解析
```

## 推荐训练顺序

```text
1. 上传 TokenHSI_g1 到服务器
2. conda activate tokenhsi_g1
3. python tools/validate_g1_interfaces.py
4. 小规模 IsaacGym smoke test: num_envs=64, horizon_length=64, minibatch_size=4096, max_iterations=1
5. 中规模短跑: num_envs=512/1024, max_iterations=50
6. 正式训练: bash tokenhsi/scripts/single_task/g1_tokenhsi_traj_train.sh
7. 根据 TensorBoard 和 rollout 现象调 reward / PD / termination
```
