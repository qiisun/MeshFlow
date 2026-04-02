#!/bin/bash
# Launch a single-node training job on TACC Vista.
# Usage: bash tacc/launch.sh <config> [overrides...] [-- sbatch args...]
#   e.g. bash tacc/launch.sh configs/overfit/smoke-min.yaml
#   e.g. bash tacc/launch.sh configs/base-120m-x1-cls.yaml train.global_batch_size=64 data.num_workers=8
#   e.g. bash tacc/launch.sh configs/base-120m-x1-cls.yaml train.global_batch_size=64 -- -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train.slurm"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config> [overrides...] [-- sbatch args...]"
    echo "  e.g. $0 configs/overfit/smoke-min.yaml"
    echo "  e.g. $0 configs/base-120m-x1-cls.yaml train.global_batch_size=64 data.num_workers=8"
    echo "  e.g. $0 configs/base-120m-x1-cls.yaml train.global_batch_size=64 -- -t 24:00:00"
    exit 1
fi

CONFIG="$1"
shift

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# Split remaining args into overrides (key=val) and sbatch args (after --)
OVERRIDES=""
SBATCH_ARGS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        SBATCH_ARGS=("$@")
        break
    fi
    OVERRIDES="${OVERRIDES} $1"
    shift
done

# Derive a readable job name from the config filename (without path/extension).
TIMESTAMP=$(date +%b%d-%I%M%p)
JOB_NAME="mf2-$(basename "${CONFIG}" .yaml)-${TIMESTAMP}"

# Build a temporary slurm script that injects the config path and overrides.
SLURM_SCRIPT=$(mktemp /tmp/meshflow_slurm_XXXXXX.sh)
sed -e "s|__CONFIG_PLACEHOLDER__|${CONFIG}|" \
    -e "s|__OVERRIDES_PLACEHOLDER__|${OVERRIDES}|" \
    -e "s|__JOB_NAME__|${JOB_NAME}|g" \
    "$TEMPLATE" > "$SLURM_SCRIPT"

echo "Submitting: config=${CONFIG}  overrides=${OVERRIDES:-none}  job_name=${JOB_NAME}"
sbatch --job-name="$JOB_NAME" "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" "$SLURM_SCRIPT"

rm -f "$SLURM_SCRIPT"
