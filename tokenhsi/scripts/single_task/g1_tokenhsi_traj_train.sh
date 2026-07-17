#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entrypoint for the original single-task command.
# Main config: tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/g1_tokenhsi_single_gpu_train.sh" "$@"
