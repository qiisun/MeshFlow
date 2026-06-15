<div align="center">
<h2>MeshFlow: Mesh Generation with Equivariant Flow Matching</h2>

<p>
  <span class="author">Qi Sun</span>,
  <span class="author"><a href="https://georgenakayama.github.io/">Kiyohiro Nakayama</a></span>,
  <span class="author"><a href="https://nathanyanjing.github.io/">Jing Nathan Yan</a></span>,
  <span class="author"><a href="https://www.cs.utexas.edu/~huangqx/">Qixing Huang</a></span>,
  <span class="author"><a href="https://rush-nlp.com/">Alexander Rush</a></span>,
  <br>
  <span class="author"><a href="https://geometry.stanford.edu/member/guibas/">Leonidas Guibas</a></span>,
  <span class="author"><a href="https://stanford.edu/~gordonwz/">Gordon Wetzstein</a></span>,
  <span class="author"><a href="https://scholar.google.com/citations?user=3s9f9VIAAAAJ&hl=zh-CN">Jing Liao</a></span>,
  <span class="author"><a href="https://www.guandaoyang.com/">Guandao Yang</a></span>
</p>

<p><b>SIGGRAPH 2026</b></p>

<p>
  <a href="https://qiisun.github.io/MeshFlow//"><img src="https://img.shields.io/badge/Project-Page-4f7cff" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/qsun2001/meshflow"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc4d" alt="Hugging Face Dataset"></a>
  <a href="https://github.com/qiisun/MeshFlow2"><img src="https://img.shields.io/badge/GitHub-Code-181717" alt="GitHub Code"></a>
  <a href="#cite"><img src="https://img.shields.io/badge/Paper-Cite-b31b1b" alt="Paper Citation"></a>
</p>
</div>

<p align="center">
  <video src="assets/0609.mp4" controls muted loop playsinline width="100%"></video>
  <br>
  <a href="https://www.youtube.com/watch?v=5950VIAiASk">Watch on YouTube</a>
</p>

MeshFlow is an unconditional mesh generation pipeline based on equivariant flow matching. This repository contains the core PyTorch training, inference, demo, rendering, and evaluation code used for mesh generation experiments.

## Installation

The code is tested with Python 3.10, PyTorch 2.4.1, CUDA 12.4, and FlashAttention 2.6.3.

```bash
conda create -n mflow python=3.10 -y
conda activate mflow
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```

Optional Chamfer extension for evaluation:

```bash
cd utils/chamfer3D
python setup.py install
cd ../..
```

## Data

Download datasets under `downloaded_data/`. For a quick overfit run, only `ss_overfit.tar.gz` is required:

```bash
mkdir -p downloaded_data
cd downloaded_data
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/ss_overfit.tar.gz
tar xf ss_overfit.tar.gz && rm ss_overfit.tar.gz
cd ..
```

Expected layout for the overfit configs:

```text
downloaded_data/ss_overfit/
  split/
    train.npz
    test.npz
  objaverse_occ_v5_ids/
    *.npz
```

<details><summary><b>Optional datasets for ShapeNet, Sketchfab, and Objaverse experiments</b></summary>

Run these commands from `downloaded_data/`:

```bash
# Sketchfab
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/sketchfab.tar.gz
tar xf sketchfab.tar.gz && rm sketchfab.tar.gz

# ShapeNet main split
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet.tar.gz
tar xf shapenet.tar.gz && rm shapenet.tar.gz

# ShapeNet class split
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet-cls.tar.gz
tar xf shapenet-cls.tar.gz && rm shapenet-cls.tar.gz

# ShapeNet rebuttal splits
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet-rebuttal.tar.gz
tar xf shapenet-rebuttal.tar.gz && rm shapenet-rebuttal.tar.gz
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet-rebuttal2.tar.gz
tar xf shapenet-rebuttal2.tar.gz && rm shapenet-rebuttal2.tar.gz

# Objaverse assets
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/objaverse_occ_v5_ids.tar.gz
wget https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/split.tar.gz
tar xf objaverse_occ_v5_ids.tar.gz && rm objaverse_occ_v5_ids.tar.gz
tar xf split.tar.gz && rm split.tar.gz
mkdir -p objaverse
mv objaverse_occ_v5_ids objaverse/
mv split objaverse/
```

</details>

## Pretrained Checkpoints (ShapeNet)

Pretrained checkpoints are hosted on the MeshFlow Hugging Face dataset repository.

| Category | Config | Checkpoint |
| --- | --- | --- |
| bench | `configs/snet/base-120m-ot-v-bench.yaml` | [`v1/120m-ot-v-bench/checkpoints/last.pt`](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-bench/checkpoints/last.pt) |
| chair | `configs/snet/base-120m-ot-v-chair.yaml` | [`v1/120m-ot-v-chair/checkpoints/last.pt`](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-chair/checkpoints/last.pt) |
| lamp | `configs/snet/base-120m-ot-v-lamp.yaml` | [`v1/120m-ot-v-lamp/checkpoints/last.pt`](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-lamp/checkpoints/last.pt) |
| table | `configs/snet/base-120m-ot-v-table.yaml` | [`v1/120m-ot-v-table/checkpoints/last.pt`](https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-table/checkpoints/last.pt) |

## Training

Overfit example:

```bash
bash tools/run_train.sh configs/overfit/base-120m-ot-x1.yaml --train.global_batch_size=4
```

ShapeNet category example:

```bash
bash tools/run_train.sh configs/snet/base-120m-x1-bench.yaml
```

Long-running ShapeNet bench example with command-line overrides:

```bash
accelerate launch \
  --num_processes 6 \
  --mixed_precision bf16 \
  --gpu_ids 0,1,2,3,4,5 \
  train.py \
  --config configs/snet/base-120m-ot-v-bench.yaml \
  train.global_batch_size=72 \
  train.max_steps=1000000 \
  train.ckpt_every=50000
```

Checkpoints are saved under `output/<exp_name>/checkpoints`. By default, training keeps the latest 3 checkpoints and runs final inference for `train.final_num_samples` meshes.

## Inference

Standalone inference example:

```bash
python inference.py \
  --config configs/overfit/base-120m-x1.yaml \
  --ckpt_path output/overfit-base-120m-x1/checkpoints/00075000.pt \
  --demo
```

ShapeNet bench inference example:

```bash
CUDA_VISIBLE_DEVICES=6 python inference.py \
  --config configs/snet/base-120m-ot-v-bench.yaml \
  --ckpt_path output/120m-ot-v-bench/checkpoints/00500000.pt \
  sample.num_samples=1000
```

Generated meshes are saved to `output/<exp_name>/infer_<step>/`. The default CFG scale is 2.0 and can be overridden with `sample.cfg_scale=<value>`.

## Interactive Demo

Launch the Gradio demo for category-conditioned single-mesh generation and post-processing:

```bash
python gradio_demo.py \
  --config configs/snet/base-120m-ot-v-bench.yaml \
  --port 7860
```

Then open `http://127.0.0.1:7860`. The UI supports category selection, checkpoint auto-switching, random seed, CFG scale, sampling steps, face count, duplicate-vertex merging, close-vertex merging, and hole filling. Each run exports `raw.obj` and `post.obj` under `output/gradio_demo/<timestamp>/`.

## Evaluation

ShapeNet generation metrics can be computed from generated `.obj` meshes:

```bash
CUDA_VISIBLE_DEVICES=6 python tools/point_evaluation.py \
  --gen-root output/120m-ot-v-bench/infer_00500000 \
  --category bench
```

Run repeated evaluations and report mean/std metrics:

```bash
CUDA_VISIBLE_DEVICES=6 python tools/point_evaluation.py \
  --gen-root output/120m-ot-v-bench/infer_00500000 \
  --category bench \
  --num-runs 5
```

Supported category names include `bench`, `bottle`, `chair`, `display`, `monitor`, `lamp`, `loudspeaker`, `speaker`, and `table`. If `--max-gen-meshes` is omitted, evaluation defaults to the number of valid test meshes for the requested category.


## Cite

Please cite our work if you find it useful:

```bibtex
@inproceedings{meshflow,
  title     = {MeshFlow: Mesh Generation with Equivariant Flow Matching},
  author    = {Sun, Qi and Nakayama, Kiyohiro and Yan, Jing Nathan and Huang, Qixing and Rush, Alexander and Guibas, Leonidas and Wetzstein, Gordon and Liao, Jing and Yang, Guandao},
  booktitle = {SIGGRAPH},
  year      = {2026}
}
```
