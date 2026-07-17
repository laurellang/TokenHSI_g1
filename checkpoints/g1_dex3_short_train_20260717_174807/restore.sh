#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUT_DIR="${REPO_ROOT}/output/g1_dex3_short_train_20260717_174807/Humanoid_17-17-48-09/nn"
OUT_FILE="${OUT_DIR}/Humanoid.pth"

mkdir -p "${OUT_DIR}"
cat "${SCRIPT_DIR}"/Humanoid.pth.part-* > "${OUT_FILE}"
echo "restored ${OUT_FILE}"
