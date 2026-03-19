# MeshFlow: Mesh Generation with Equivariant Flow Matching

This repository contains the training, sampling, and evaluation code for MeshFlow2.
The current release focuses on the **EquiDiT mesh pipeline** (`model_type: equidit`) and keeps the workflow minimal and reproducible.

---

## 1) What is included

- Training entrypoint: `train_pixel.py`
- Sampling entrypoint: `inference_dit.py` (with compatibility wrapper `inference.py`)
- Standalone evaluation sampling script: `tools/infer_mesh_equidit.py`
- Point-based metrics: `tools/point_evaluation.py`
- Main train launcher: `tools/run_train.sh`
- Configs: `configs/overfit`, `configs/rebuttal`, `configs/base_jit.yaml`

---

## 2) Environment setup

### 2.1 Create environment

```bash
conda create -n mflow python=3.10 -y
conda activate mflow

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2.2 (Optional but recommended) install chamfer extension

Chamfer is used in validation/evaluation if available.

```bash
cd utils/chamfer3D
python setup.py install
cd ../..
```

If this step is skipped, training still runs, but Chamfer-related metrics are skipped with a warning.

---

## 3) Data preparation

Create data root:

```bash
mkdir -p downloaded_data
cd downloaded_data
```

### 3.1 Overfit benchmark set

```bash
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/ss_overfit.tar.gz
tar xf ss_overfit.tar.gz
rm ss_overfit.tar.gz
```

### 3.2 ShapeNet splits (for category-level runs)

```bash
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-rebuttal.tar.gz
tar xf shapenet-rebuttal.tar.gz
rm shapenet-rebuttal.tar.gz
```

### 3.3 Sketchfab (optional)

```bash
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/sketchfab.tar.gz
tar xf sketchfab.tar.gz
rm sketchfab.tar.gz
```

### 3.4 Full dataset download list (camera-ready completeness)

If you want the full set used across different experiments (instead of only the quick-start subsets),
you can run the commands below.

```bash
# dummy
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/dummy.tar.gz
tar xf dummy.tar.gz
rm dummy.tar.gz

# ss_overfit
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/ss_overfit.tar.gz
tar xf ss_overfit.tar.gz
rm ss_overfit.tar.gz

# shapenet
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet.tar.gz
tar xf shapenet.tar.gz
rm shapenet.tar.gz

# sketchfab
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/sketchfab.tar.gz
tar xf sketchfab.tar.gz
rm sketchfab.tar.gz

# objaverse_occ_v5_ids + split (reorganized into downloaded_data/objaverse)
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/objaverse_occ_v5_ids.tar.gz
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/split.tar.gz
tar xf objaverse_occ_v5_ids.tar.gz
tar xf split.tar.gz
rm objaverse_occ_v5_ids.tar.gz
rm split.tar.gz
mkdir -p objaverse
mv objaverse_occ_v5_ids objaverse/
mv split objaverse/

# extra shapenet assets
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-cls.tar.gz
tar xf shapenet-cls.tar.gz
rm shapenet-cls.tar.gz

wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-rebuttal2.tar.gz
tar xf shapenet-rebuttal2.tar.gz
rm shapenet-rebuttal2.tar.gz

wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-rebuttal.tar.gz
tar xf shapenet-rebuttal.tar.gz
rm shapenet-rebuttal.tar.gz
```

Back to repo root:

```bash
cd ..
```

> Expected dataset format for this codebase: each dataset root contains `split/*.npz` and mesh npz files under `objaverse_occ_v5_ids/`.

---

## 4) Quick start (recommended commands)

### 4.1 Training

```bash
# 500M baseline
bash tools/run_train.sh configs/overfit/base-500m.yaml

# 500M + OT + JIT x1
bash tools/run_train.sh configs/overfit/base-500m-ot-x1.yaml

# 120M baseline
bash tools/run_train.sh configs/overfit/base-120m.yaml

# 120M + OT + JIT x1
bash tools/run_train.sh configs/overfit/base-120m-ot-x1.yaml

# 120M + JIT x1
bash tools/run_train.sh configs/overfit/base-120m-x1.yaml

# Rebuttal config
bash tools/run_train.sh configs/rebuttal/base-120m-x1.yaml
```

### 4.2 Sampling from a checkpoint

```bash
python inference_dit.py --config configs/overfit/base-120m-ot-x1.yaml
```

`ckpt_path` must be set in the config, or provided through your workflow.

---

## 5) Standalone inference + automatic point evaluation

Use `tools/infer_mesh_equidit.py` for repeated sampling and summary metrics.

```bash
python tools/infer_mesh_equidit.py \
  --config configs/rebuttal/base-120m-x1.yaml \
  --num-samples 100 \
  --batch-size 16 \
  --use-test-faces
```

Key behavior:

- Auto-resolves latest checkpoint under `train.output_dir/train.exp_name/checkpoints` (if `--ckpt` is omitted)
- Saves meshes under `infer_<ckpt>_eval`
- Runs multiple repeats (`--num-runs`, default `5`) and writes averaged summary
- Saves run-level metrics to `point_metrics.json`
- Saves aggregated metrics to `point_metrics_summary.json`

Useful overrides:

- `--ckpt`: explicit checkpoint path
- `--out-dir`: explicit output path
- `--cfg-scale`, `--num-steps`: sampling controls
- `--test-data-path`: evaluate against another dataset root

---

## 6) Current model support

- Supported model type: `equidit`
- Removed from this release: `dit_llama` path

If a config sets another `model_type`, training/inference now raises a clear error.

---

## 7) Notes on Chamfer during training

Validation Chamfer is computed when:

1. Chamfer extension is installed (`utils/chamfer3D`), and
2. The run is recognized as overfit-style evaluation (or explicitly enabled in config).

You can explicitly control this in config:

```yaml
sample:
  compute_chamfer: true
```

---

## 8) Reproducibility tips

- Keep `configs/*` under version control for each run.
- Log exact commit hash together with checkpoint path.
- Avoid editing `tools/run_train.sh` mid-experiment; clone and freeze a launcher per experiment if needed.

---

## 9) Troubleshooting

### Training exits early / launch issues

- Check GPU IDs in `tools/run_train.sh` (`TARGET_GPU_ID`, `GPUS_PER_NODE`).
- Ensure `accelerate` can see the same number of GPUs as configured.

### No Chamfer shown in logs

- Install `utils/chamfer3D` extension.
- Ensure overfit evaluation is active or set `sample.compute_chamfer: true`.

### Checkpoint not found in standalone inference

- Verify `train.output_dir` + `train.exp_name` in config.
- Or pass `--ckpt` manually.

---

## 10) Repository structure (minimal)

```text
train_pixel.py                # training
inference_dit.py              # sampling
inference.py                  # wrapper
configs/
  overfit/
  rebuttal/
  base_jit.yaml
tools/
  run_train.sh
  infer_mesh_equidit.py
  point_evaluation.py
datasets/
models/
transport/
utils/
```

---

If you use this codebase for your camera-ready artifact, we recommend keeping this README and the exact used configs in the release package as-is for reproducibility.
