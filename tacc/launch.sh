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

# Split remaining args: anything containing '=' is an override, '--' separates sbatch args.
OVERRIDES=""
SBATCH_ARGS=""
PAST_SEP=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        PAST_SEP=true
        continue
    fi
    if $PAST_SEP; then
        SBATCH_ARGS="${SBATCH_ARGS} ${arg}"
    else
        OVERRIDES="${OVERRIDES} ${arg}"
    fi
done

# Derive a readable job name from the config filename (without path/extension).
TIMESTAMP=$(date +%b%d-%I%M%p)
JOB_NAME="mf2-$(basename "${CONFIG}" .yaml)-${TIMESTAMP}"

# Auto-inject a unique train.exp_name so each submission writes to its own output dir,
# unless the caller already provided one.
if ! echo "$OVERRIDES" | grep -q "train.exp_name="; then
    OVERRIDES="${OVERRIDES} train.exp_name=${JOB_NAME}"
fi

# Build a temporary slurm script that injects the config path and overrides.
SLURM_SCRIPT=$(mktemp /tmp/meshflow_slurm_XXXXXX.sh)
sed -e "s|__CONFIG_PLACEHOLDER__|${CONFIG}|" \
    -e "s|__OVERRIDES_PLACEHOLDER__|${OVERRIDES}|" \
    -e "s|__JOB_NAME__|${JOB_NAME}|g" \
    "$TEMPLATE" > "$SLURM_SCRIPT"

echo "Submitting: config=${CONFIG}  overrides=${OVERRIDES:-none}  job_name=${JOB_NAME}"
echo "Generated script:"
cat "$SLURM_SCRIPT" | grep "accelerate launch"
sbatch --job-name="$JOB_NAME" ${SBATCH_ARGS} "$SLURM_SCRIPT"

rm -f "$SLURM_SCRIPT"
