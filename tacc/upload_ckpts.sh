#!/usr/bin/env bash
# Upload every checkpoint under one or more experiment folders to HF.
#
# Each <exp_dir> is an experiment folder, e.g.:   output/<exp_name>
# Checkpoints are read from:                      <exp_dir>/checkpoints/*.pt
# Files land at:                                  datasets/<repo>/<prefix>/<exp_name>/checkpoints/<file>
#
# Usage:
#   bash tacc/upload_ckpts.sh <exp_dir> [<exp_dir> ...]
#   REPO=guandao/meshflow_checkpoints PREFIX=v1 bash tacc/upload_ckpts.sh output/run_a output/run_b

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <exp_dir> [<exp_dir> ...]" >&2
    echo "  e.g. $0 output/base-120m output/base-500m" >&2
    exit 1
fi

: "${HF_TOKEN:?HF_TOKEN is not set. export HF_TOKEN=hf_... first.}"
REPO="${REPO:-guandao/meshflow_checkpoints}"
PREFIX="${PREFIX:-v1}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${REPO_ROOT}/tools/upload_ckpt_to_hf.py"

for EXP_DIR in "$@"; do
    if [ ! -d "$EXP_DIR" ]; then
        echo "[skip] not a directory: $EXP_DIR" >&2
        continue
    fi

    CKPT_DIR="${EXP_DIR%/}/checkpoints"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "[skip] no checkpoints/ subdir in $EXP_DIR" >&2
        continue
    fi

    EXP="$(basename "$(cd "$EXP_DIR" && pwd)")"

    mapfile -t CKPTS < <(find "$CKPT_DIR" -maxdepth 1 -type f -name '*.pt' | sort)
    if [ "${#CKPTS[@]}" -eq 0 ]; then
        echo "[skip] no .pt files in $CKPT_DIR" >&2
        continue
    fi

    echo "=== $EXP : ${#CKPTS[@]} checkpoint(s) from $CKPT_DIR ==="
    for CKPT in "${CKPTS[@]}"; do
        FILE="$(basename "$CKPT")"
        TARGET="${PREFIX}/${EXP}/checkpoints/${FILE}"
        echo "  -> $TARGET"
        python "$PY" \
            --source "$CKPT" \
            --repo-id "$REPO" \
            --path-in-repo "$TARGET" \
            --commit-message "Upload ${EXP}/${FILE}"
    done
done

echo "Done."
