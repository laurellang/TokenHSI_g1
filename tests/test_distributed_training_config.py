from types import SimpleNamespace
from pathlib import Path
import subprocess

import pytest

from tokenhsi.utils.distributed import HorovodRuntime, apply_horovod_runtime


ROOT = Path(__file__).resolve().parents[1]


class FakeHorovod:
    def __init__(self, rank=3, local_rank=1, size=8, local_size=4):
        self._rank = rank
        self._local_rank = local_rank
        self._size = size
        self._local_size = local_size

    def rank(self):
        return self._rank

    def local_rank(self):
        return self._local_rank

    def size(self):
        return self._size

    def local_size(self):
        return self._local_size


def test_horovod_runtime_uses_local_rank_for_cuda_device_and_rank_for_seed():
    runtime = HorovodRuntime.from_horovod(FakeHorovod(), base_seed=100, num_envs_per_rank=4096)

    assert runtime.rank == 3
    assert runtime.local_rank == 1
    assert runtime.world_size == 8
    assert runtime.local_world_size == 4
    assert runtime.seed == 103
    assert runtime.device == "cuda"
    assert runtime.rl_device == "cuda:1"
    assert runtime.num_envs_per_rank == 4096
    assert runtime.global_num_envs == 32768


def test_apply_horovod_runtime_updates_args_and_configs_in_place():
    args = SimpleNamespace(device="cpu", device_id=0, rl_device="cuda:0")
    cfg = {"env": {"numEnvs": 128}}
    cfg_train = {"params": {"seed": 42, "config": {"num_actors": 128}}}
    runtime = HorovodRuntime.from_horovod(FakeHorovod(rank=7, local_rank=7), base_seed=42, num_envs_per_rank=4096)

    apply_horovod_runtime(runtime, args, cfg, cfg_train)

    assert args.device == "cuda"
    assert args.device_id == 7
    assert args.rl_device == "cuda:7"
    assert cfg["rank"] == 7
    assert cfg["local_rank"] == 7
    assert cfg["world_size"] == 8
    assert cfg["rl_device"] == "cuda:7"
    assert cfg["env"]["numEnvs"] == 4096
    assert cfg_train["params"]["seed"] == 49
    assert cfg_train["params"]["config"]["seed"] == 49
    assert cfg_train["params"]["config"]["num_actors"] == 4096
    assert cfg_train["params"]["config"]["global_num_actors"] == 32768


def test_horovod_runtime_rejects_non_positive_envs_per_rank():
    with pytest.raises(ValueError, match="num_envs_per_rank"):
        HorovodRuntime.from_horovod(FakeHorovod(), base_seed=1, num_envs_per_rank=0)


def test_8gpu_launcher_defaults_to_512_envs_per_gpu_and_mpirun_horovod():
    launcher = ROOT / "tokenhsi/scripts/multi_gpu/g1_tokenhsi_8gpu_train.sh"
    text = launcher.read_text()

    assert "NUM_GPUS=${NUM_GPUS:-8}" in text
    assert "HOROVOD_HOSTS=${HOROVOD_HOSTS:-localhost:${NUM_GPUS}}" in text
    assert "DISTRIBUTED_LAUNCHER=${DISTRIBUTED_LAUNCHER:-mpirun}" in text
    assert "NUM_ENVS_PER_GPU=${NUM_ENVS_PER_GPU:-512}" in text
    assert "mpirun" in text
    assert "-np \"${NUM_GPUS}\"" in text
    assert "--host \"${HOROVOD_HOSTS}\"" in text
    assert "--bind-to none" in text
    assert "--map-by slot" in text
    assert "--horovod" in text
    assert "--num_envs \"${NUM_ENVS_PER_GPU}\"" in text


def test_single_gpu_launcher_defaults_to_4096_envs_without_horovod():
    launcher = ROOT / "tokenhsi/scripts/single_task/g1_tokenhsi_single_gpu_train.sh"
    text = launcher.read_text()

    assert "GPU_ID=${GPU_ID:-}" in text
    assert "export CUDA_VISIBLE_DEVICES=\"${GPU_ID}\"" in text
    assert "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-all}" in text
    assert "NUM_ENVS=${NUM_ENVS:-4096}" in text
    assert "python ./tokenhsi/run.py" in text
    assert "--num_envs \"${NUM_ENVS}\"" in text
    assert "--headless" in text
    assert "--horovod" not in text
    assert "horovodrun" not in text


def test_single_gpu_launcher_dry_run_autofixes_small_smoke_batch_contract():
    launcher = ROOT / "tokenhsi/scripts/single_task/g1_tokenhsi_single_gpu_train.sh"

    result = subprocess.run(
        [
            "bash",
            str(launcher),
        ],
        cwd=str(ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "NUM_ENVS": "64",
            "MAX_ITERATIONS": "1",
            "DRY_RUN": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    assert "envs: 64" in result.stdout
    assert "horizon: 64" in result.stdout
    assert "minibatch: 4096" in result.stdout
    assert "--horizon_length 64" in result.stdout
    assert "--minibatch_size 4096" in result.stdout


def test_multi_gpu_launcher_dry_run_uses_mpirun_with_explicit_host_slots():
    launcher = ROOT / "tokenhsi/scripts/multi_gpu/g1_tokenhsi_8gpu_train.sh"

    result = subprocess.run(
        [
            "bash",
            str(launcher),
        ],
        cwd=str(ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "NUM_GPUS": "7",
            "CUDA_VISIBLE_DEVICES": "1,2,3,4,5,6,7",
            "DRY_RUN": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    assert "GPUs: 7" in result.stdout
    assert "hosts: localhost:7" in result.stdout
    assert "launcher: mpirun" in result.stdout
    assert "mpirun -np 7 --host localhost:7 --bind-to none --map-by slot" in result.stdout


def test_multi_gpu_launcher_dry_run_exports_python_lib_for_isaacgym():
    launcher = ROOT / "tokenhsi/scripts/multi_gpu/g1_tokenhsi_8gpu_train.sh"

    result = subprocess.run(
        [
            "bash",
            str(launcher),
        ],
        cwd=str(ROOT),
        env={
            "PATH": "/usr/bin:/bin",
            "NUM_GPUS": "2",
            "PYTHON_BIN": "/opt/conda/envs/tokenhsi_g1/bin/python",
            "DRY_RUN": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    assert "python lib: /opt/conda/envs/tokenhsi_g1/lib" in result.stdout
    assert "LD_LIBRARY_PATH=/opt/conda/envs/tokenhsi_g1/lib:" in result.stdout
