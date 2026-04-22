#!/bin/bash
# Launch a single-node inference job on TACC Vista.
# Usage: bash tacc/infer.sh <config> <ckpt_path> [overrides...] [-- sbatch args...]
#   e.g. bash tacc/infer.sh configs/overfit/base-500m-ot-x1.yaml output/mf2-.../checkpoints/00050000.pt
#   e.g. bash tacc/infer.sh configs/overfit/base-500m-ot-x1.yaml output/.../00050000.pt sample.cfg_scale=5.0
#   e.g. bash tacc/infer.sh configs/overfit/base-500m-ot-x1.yaml output/.../00050000.pt -- -t 02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/infer.slurm"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config> <ckpt_path> [overrides...] [-- sbatch args...]"
    echo "  e.g. $0 configs/overfit/base-500m-ot-x1.yaml output/mf2-.../checkpoints/00050000.pt"
    echo "  e.g. $0 configs/overfit/base-500m-ot-x1.yaml output/.../00050000.pt sample.cfg_scale=5.0"
    exit 1
fi

CONFIG="$1"
CKPT="$2"
shift 2

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "Warning: checkpoint file not found at $CKPT (continuing; may exist on compute node)"
fi

# Split remaining args: anything before '--' is an override, after is sbatch args.
OVERRIDES="ckpt_path=${CKPT}"
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

# Derive a readable job name from config + checkpoint step.
TIMESTAMP=$(date +%b%d-%I%M%p)
CKPT_STEM=$(basename "${CKPT}" .pt)
JOB_NAME="mf2-infer-$(basename "${CONFIG}" .yaml)-${CKPT_STEM}-${TIMESTAMP}"

# Build a temporary slurm script that injects the config path and overrides.
SLURM_SCRIPT=$(mktemp /tmp/meshflow_infer_XXXXXX.sh)
sed -e "s|__CONFIG_PLACEHOLDER__|${CONFIG}|" \
    -e "s|__OVERRIDES_PLACEHOLDER__|${OVERRIDES}|" \
    -e "s|__JOB_NAME__|${JOB_NAME}|g" \
    "$TEMPLATE" > "$SLURM_SCRIPT"

echo "Submitting: config=${CONFIG}  ckpt=${CKPT}  overrides=${OVERRIDES}  job_name=${JOB_NAME}"
echo "Generated script:"
cat "$SLURM_SCRIPT" | grep "python inference.py"
sbatch --job-name="$JOB_NAME" ${SBATCH_ARGS} "$SLURM_SCRIPT"

rm -f "$SLURM_SCRIPT"
