#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS=${NUM_ENVS:-4096}
GPU_ID=${GPU_ID:-}
HORIZON_LENGTH_WAS_SET=${HORIZON_LENGTH+x}
MINIBATCH_SIZE_WAS_SET=${MINIBATCH_SIZE+x}
HORIZON_LENGTH=${HORIZON_LENGTH:-32}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
MIN_ROLLOUT_BATCH=${MIN_ROLLOUT_BATCH:-4096}
MAX_ITERATIONS=${MAX_ITERATIONS:-0}
OUTPUT_PATH=${OUTPUT_PATH:-output/g1_dex3_single_gpu_${NUM_ENVS}env}
DRY_RUN=${DRY_RUN:-0}

if (( NUM_ENVS <= 0 )); then
  echo "NUM_ENVS must be positive, got ${NUM_ENVS}" >&2
  exit 2
fi

if (( HORIZON_LENGTH <= 0 )); then
  echo "HORIZON_LENGTH must be positive, got ${HORIZON_LENGTH}" >&2
  exit 2
fi

if (( MINIBATCH_SIZE <= 0 )); then
  echo "MINIBATCH_SIZE must be positive, got ${MINIBATCH_SIZE}" >&2
  exit 2
fi

ROLLOUT_BATCH=$((NUM_ENVS * HORIZON_LENGTH))
if [[ -z "${HORIZON_LENGTH_WAS_SET}" ]] && (( ROLLOUT_BATCH < MIN_ROLLOUT_BATCH )); then
  HORIZON_LENGTH=$(((MIN_ROLLOUT_BATCH + NUM_ENVS - 1) / NUM_ENVS))
  ROLLOUT_BATCH=$((NUM_ENVS * HORIZON_LENGTH))
fi

if [[ -z "${MINIBATCH_SIZE_WAS_SET}" ]] && { (( MINIBATCH_SIZE > ROLLOUT_BATCH )) || (( ROLLOUT_BATCH % MINIBATCH_SIZE != 0 )); }; then
  MINIBATCH_SIZE="${ROLLOUT_BATCH}"
fi

if (( ROLLOUT_BATCH < MIN_ROLLOUT_BATCH )); then
  echo "rollout batch NUM_ENVS*HORIZON_LENGTH must be at least ${MIN_ROLLOUT_BATCH}, got ${ROLLOUT_BATCH}" >&2
  echo "Try HORIZON_LENGTH=$(((MIN_ROLLOUT_BATCH + NUM_ENVS - 1) / NUM_ENVS)) for NUM_ENVS=${NUM_ENVS}." >&2
  exit 2
fi

if (( MINIBATCH_SIZE > ROLLOUT_BATCH )) || (( ROLLOUT_BATCH % MINIBATCH_SIZE != 0 )); then
  echo "MINIBATCH_SIZE must divide rollout batch NUM_ENVS*HORIZON_LENGTH=${ROLLOUT_BATCH}, got ${MINIBATCH_SIZE}" >&2
  exit 2
fi

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

export PYTHONPATH=".:tokenhsi:${PYTHONPATH:-}"

CMD=(
  python ./tokenhsi/run.py
  --task HumanoidTrajSitCarryClimb
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml
  --cfg_env tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml
  --motion_file tokenhsi/data/dataset_g1_all.yaml
  --num_envs "${NUM_ENVS}"
  --horizon_length "${HORIZON_LENGTH}"
  --minibatch_size "${MINIBATCH_SIZE}"
  --headless
  --output_path "${OUTPUT_PATH}"
)

if [[ "${MAX_ITERATIONS}" != "0" ]]; then
  CMD+=(--max_iterations "${MAX_ITERATIONS}")
fi

echo "Launching TokenHSI G1 single-GPU training"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-all}"
echo "  envs: ${NUM_ENVS}"
echo "  horizon: ${HORIZON_LENGTH}"
echo "  minibatch: ${MINIBATCH_SIZE}"
echo "  rollout batch: ${ROLLOUT_BATCH}"
echo "  output: ${OUTPUT_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "Command:"
  printf " %s" "${CMD[@]}"
  printf "\n"
  exit 0
fi

"${CMD[@]}"
