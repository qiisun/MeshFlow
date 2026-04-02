#!/bin/bash
# Launch a single-node training job on TACC Vista.
# Usage: bash tacc/launch.sh <config> [sbatch args...]
#   e.g. bash tacc/launch.sh configs/overfit/smoke-min.yaml
#   e.g. bash tacc/launch.sh configs/base-120m-x1-cls.yaml -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train.slurm"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config> [sbatch args...]"
    echo "  e.g. $0 configs/overfit/smoke-min.yaml"
    echo "  e.g. $0 configs/base-120m-x1-cls.yaml -t 24:00:00"
    exit 1
fi

CONFIG="$1"
shift

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# Derive a readable job name from the config filename (without path/extension).
JOB_NAME="mf2-$(basename "${CONFIG}" .yaml)"

# Build a temporary slurm script that injects the config path.
SLURM_SCRIPT=$(mktemp /tmp/meshflow_slurm_XXXXXX.sh)
sed "s|__CONFIG_PLACEHOLDER__|${CONFIG}|" "$TEMPLATE" > "$SLURM_SCRIPT"

echo "Submitting: config=${CONFIG}  job_name=${JOB_NAME}"
sbatch --job-name="$JOB_NAME" "$@" "$SLURM_SCRIPT"

rm -f "$SLURM_SCRIPT"
