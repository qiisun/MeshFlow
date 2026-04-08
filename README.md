# MeshFlow: Mesh Generation with Equivariant Flow Matching

This repository is trimmed to a minimal training/inference/evaluation pipeline for unconditional mesh generation.

## Kept Entry Points

- Training: `train.py`
- Inference: `inference.py`
- Train launcher: `tools/run_train.sh`
- Chamfer curve eval (for overfitting experiments): `tools/plot_chamfer_vs_steps.py`
- Generation metrics (for ShapeNet category, e.g. 1-NNA): `tools/point_evaluation.py`
- flow matching core: `flow_matching.py`


## Minimal Config Set

- `configs/base_jit.yaml`
- `configs/overfit/base-120m-ot.yaml`
- `configs/overfit/base-120m-ot-x1.yaml`
- `configs/overfit/base-120m-x1.yaml`

## Quick Start

### 1) Environment

```bash
conda create -n mflow python=3.10 -y
conda activate mflow
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```

Optional Chamfer extension:

```bash
cd utils/chamfer3D
python setup.py install
cd ../..
```

### 2) Train
```bash
bash tools/run_train.sh configs/overfit/base-120m-ot-x1.yaml --train.global_batch_size=4
```

### 3) Inference (standalone)

```bash
python inference.py --config configs/overfit/base-120m-x1.yaml --ckpt_path=output/overfit-base-120m-x1/checkpoints/00075000.pt --use_qk_norm=false --demo # --demo means only generate 8 samples
```

### 4) Generation Metrics Evaluation [TODO]

```bash
python tools/point_evaluation.py --help
```

## Dataset Preparation

Run all commands from repository root.

```bash
mkdir -p downloaded_data
cd downloaded_data
```

### Required for overfit configs

```bash
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/ss_overfit.tar.gz
tar xf ss_overfit.tar.gz
rm ss_overfit.tar.gz
```

### Optional datasets for other experiments

```bash
# sketchfab
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/sketchfab.tar.gz
tar xf sketchfab.tar.gz
rm sketchfab.tar.gz

# shapenet (main)
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet.tar.gz
tar xf shapenet.tar.gz
rm shapenet.tar.gz

# shapenet class split
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-cls.tar.gz
tar xf shapenet-cls.tar.gz
rm shapenet-cls.tar.gz

# shapenet rebuttal splits
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-rebuttal.tar.gz
tar xf shapenet-rebuttal.tar.gz
rm shapenet-rebuttal.tar.gz

wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-rebuttal2.tar.gz
tar xf shapenet-rebuttal2.tar.gz
rm shapenet-rebuttal2.tar.gz

# objaverse assets
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/objaverse_occ_v5_ids.tar.gz
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/split.tar.gz
tar xf objaverse_occ_v5_ids.tar.gz
tar xf split.tar.gz
rm objaverse_occ_v5_ids.tar.gz
rm split.tar.gz
mkdir -p objaverse
mv objaverse_occ_v5_ids objaverse/
mv split objaverse/
```

Back to repository root:

```bash
cd ..
```

Expected structure for overfit training:

```text
downloaded_data/ss_overfit/
  split/
    train.npz
    test.npz
  objaverse_occ_v5_ids/
    *.npz
```

## Train ShapeNet Category

```bash
bash tools/run_train.sh configs/snet/base-120m-x1-bench.yaml
```


## Structure

```text
train_pixel.py
inference_dit.py
transport_simple.py
configs/
  base_jit.yaml
  overfit/
tools/
  run_train.sh
  plot_chamfer_vs_steps.py
  point_evaluation.py
models/
datasets/
utils/
```


## Citations
Please cite our work if you find it useful:

```latex
@inproceedings{meshflow,
 title={},
 author={},
 journal={},
 year={2026}
}
```