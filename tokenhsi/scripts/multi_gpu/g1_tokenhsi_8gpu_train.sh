#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
HOROVOD_HOSTS=${HOROVOD_HOSTS:-localhost:${NUM_GPUS}}
DISTRIBUTED_LAUNCHER=${DISTRIBUTED_LAUNCHER:-mpirun}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_ENVS_PER_GPU=${NUM_ENVS_PER_GPU:-512}
HORIZON_LENGTH=${HORIZON_LENGTH:-32}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
MAX_ITERATIONS=${MAX_ITERATIONS:-0}
OUTPUT_PATH=${OUTPUT_PATH:-output/g1_dex3_hvd_${NUM_GPUS}gpu_${NUM_ENVS_PER_GPU}env}
DRY_RUN=${DRY_RUN:-0}

export PYTHONPATH=".:tokenhsi:${PYTHONPATH:-}"
PYTHON_BIN_DIR="$(dirname "${PYTHON_BIN}")"
PYTHON_LIB_DIR="$(dirname "${PYTHON_BIN_DIR}")/lib"
export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"

CMD=(
  "${PYTHON_BIN}" ./tokenhsi/run.py
  --task HumanoidTrajSitCarryClimb
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml
  --cfg_env tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml
  --motion_file tokenhsi/data/dataset_g1_all.yaml
  --num_envs "${NUM_ENVS_PER_GPU}"
  --horizon_length "${HORIZON_LENGTH}"
  --minibatch_size "${MINIBATCH_SIZE}"
  --headless
  --horovod
  --output_path "${OUTPUT_PATH}"
)

if [[ "${MAX_ITERATIONS}" != "0" ]]; then
  CMD+=(--max_iterations "${MAX_ITERATIONS}")
fi

case "${DISTRIBUTED_LAUNCHER}" in
  mpirun)
    LAUNCH_CMD=(
      mpirun
      -np "${NUM_GPUS}"
      --host "${HOROVOD_HOSTS}"
      --bind-to none
      --map-by slot
      -x CUDA_VISIBLE_DEVICES
      -x PYTHONPATH
      -x LD_LIBRARY_PATH
      -x PATH
    )
    ;;
  horovodrun)
    LAUNCH_CMD=(
      horovodrun
      -np "${NUM_GPUS}"
      -H "${HOROVOD_HOSTS}"
    )
    ;;
  *)
    echo "DISTRIBUTED_LAUNCHER must be 'mpirun' or 'horovodrun', got '${DISTRIBUTED_LAUNCHER}'" >&2
    exit 2
    ;;
esac

FINAL_CMD=("${LAUNCH_CMD[@]}" "${CMD[@]}")

echo "Launching TokenHSI G1 distributed training"
echo "  GPUs: ${NUM_GPUS}"
echo "  hosts: ${HOROVOD_HOSTS}"
echo "  launcher: ${DISTRIBUTED_LAUNCHER}"
echo "  envs/GPU: ${NUM_ENVS_PER_GPU}"
echo "  global envs: $((NUM_GPUS * NUM_ENVS_PER_GPU))"
echo "  horizon: ${HORIZON_LENGTH}"
echo "  minibatch: ${MINIBATCH_SIZE}"
echo "  output: ${OUTPUT_PATH}"
echo "  python: ${PYTHON_BIN}"
echo "  python lib: ${PYTHON_LIB_DIR}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "Command:"
  printf " %s" "${FINAL_CMD[@]}"
  printf "\n"
  exit 0
fi

"${FINAL_CMD[@]}"
