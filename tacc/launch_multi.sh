#!/bin/bash
# Launch a multi-node training job on TACC Vista (gh partition: 1 GPU per node).
# Usage: bash tacc/launch_multi.sh <num_nodes> <config> [overrides...] [-- sbatch args...]
#   e.g. bash tacc/launch_multi.sh 4 configs/base-120m-x1-cls.yaml
#   e.g. bash tacc/launch_multi.sh 8 configs/base_ordered.yaml train.global_batch_size=256
#   e.g. bash tacc/launch_multi.sh 4 configs/base_ordered.yaml -- -t 24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train_multi.slurm"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <num_nodes> <config> [overrides...] [-- sbatch args...]"
    echo "  e.g. $0 4 configs/base-120m-x1-cls.yaml"
    echo "  e.g. $0 8 configs/base_ordered.yaml train.global_batch_size=256"
    echo "  e.g. $0 4 configs/base_ordered.yaml -- -t 24:00:00"
    exit 1
fi

NUM_NODES="$1"
shift
CONFIG="$1"
shift

if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]] || [ "$NUM_NODES" -lt 1 ]; then
    echo "Error: num_nodes must be a positive integer, got: $NUM_NODES"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# Split remaining args: anything before '--' is an override, after is sbatch args.
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
JOB_NAME="mf2-n${NUM_NODES}-$(basename "${CONFIG}" .yaml)-${TIMESTAMP}"

# Auto-inject a unique train.exp_name so each submission writes to its own output dir,
# unless the caller already provided one.
if ! echo "$OVERRIDES" | grep -q "train.exp_name="; then
    OVERRIDES="${OVERRIDES} train.exp_name=${JOB_NAME}"
fi

# Build a temporary slurm script that injects num_nodes, config path, and overrides.
SLURM_SCRIPT=$(mktemp /tmp/meshflow_slurm_multi_XXXXXX.sh)
sed -e "s|__NUM_NODES__|${NUM_NODES}|g" \
    -e "s|__CONFIG_PLACEHOLDER__|${CONFIG}|" \
    -e "s|__OVERRIDES_PLACEHOLDER__|${OVERRIDES}|" \
    -e "s|__JOB_NAME__|${JOB_NAME}|g" \
    "$TEMPLATE" > "$SLURM_SCRIPT"

echo "Submitting: nodes=${NUM_NODES}  config=${CONFIG}  overrides=${OVERRIDES:-none}  job_name=${JOB_NAME}"
echo "Generated launch line:"
grep "accelerate launch" "$SLURM_SCRIPT"
sbatch --job-name="$JOB_NAME" ${SBATCH_ARGS} "$SLURM_SCRIPT"

rm -f "$SLURM_SCRIPT"
