from dataclasses import dataclass


@dataclass(frozen=True)
class HorovodRuntime:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    seed: int
    device: str
    rl_device: str
    num_envs_per_rank: int
    global_num_envs: int

    @classmethod
    def from_horovod(cls, hvd, base_seed, num_envs_per_rank):
        if num_envs_per_rank <= 0:
            raise ValueError("num_envs_per_rank must be positive, got {}".format(num_envs_per_rank))

        rank = int(hvd.rank())
        local_rank = int(hvd.local_rank())
        world_size = int(hvd.size())
        local_world_size = int(hvd.local_size())

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            seed=int(base_seed) + rank,
            device="cuda",
            rl_device="cuda:{}".format(local_rank),
            num_envs_per_rank=int(num_envs_per_rank),
            global_num_envs=int(num_envs_per_rank) * world_size,
        )


def ensure_horovod_initialized(hvd):
    is_initialized = getattr(hvd, "is_initialized", None)
    if is_initialized is not None:
        if not is_initialized():
            hvd.init()
        return

    try:
        hvd.init()
    except ValueError as exc:
        if "already" not in str(exc).lower() and "initialized" not in str(exc).lower():
            raise


def apply_horovod_runtime(runtime, args, cfg, cfg_train):
    args.device = runtime.device
    args.device_id = runtime.local_rank
    args.rl_device = runtime.rl_device

    cfg["rank"] = runtime.rank
    cfg["local_rank"] = runtime.local_rank
    cfg["world_size"] = runtime.world_size
    cfg["local_world_size"] = runtime.local_world_size
    cfg["rl_device"] = runtime.rl_device
    cfg["env"]["numEnvs"] = runtime.num_envs_per_rank

    cfg_train["params"]["seed"] = runtime.seed
    cfg_train["params"]["config"]["seed"] = runtime.seed
    cfg_train["params"]["config"]["num_actors"] = runtime.num_envs_per_rank
    cfg_train["params"]["config"]["global_num_actors"] = runtime.global_num_envs
    cfg_train["params"]["config"]["world_size"] = runtime.world_size
    cfg_train["params"]["config"]["local_world_size"] = runtime.local_world_size

    return cfg, cfg_train
