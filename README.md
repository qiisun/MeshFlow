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
</div>

<p align="center">
  <img src="assets/teaser-meshflow.gif" alt="MeshFlow teaser" width="100%"/>
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
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/ss_overfit.tar.gz
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
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/sketchfab.tar.gz
tar xf sketchfab.tar.gz && rm sketchfab.tar.gz

# ShapeNet main split
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet.tar.gz
tar xf shapenet.tar.gz && rm shapenet.tar.gz

# ShapeNet class split
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-cls.tar.gz
tar xf shapenet-cls.tar.gz && rm shapenet-cls.tar.gz

# ShapeNet rebuttal splits
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-rebuttal.tar.gz
tar xf shapenet-rebuttal.tar.gz && rm shapenet-rebuttal.tar.gz
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-rebuttal2.tar.gz
tar xf shapenet-rebuttal2.tar.gz && rm shapenet-rebuttal2.tar.gz

# Objaverse assets
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/objaverse_occ_v5_ids.tar.gz
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/split.tar.gz
tar xf objaverse_occ_v5_ids.tar.gz && rm objaverse_occ_v5_ids.tar.gz
tar xf split.tar.gz && rm split.tar.gz
mkdir -p objaverse
mv objaverse_occ_v5_ids objaverse/
mv split objaverse/
```

</details>

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

## Rendering Tools

Render a folder of `.obj`/`.ply` meshes to PNG previews for visual picking:

```bash
blender -b -P tools/render_mesh_folder.py -- \
  --input output/render-pipeline-denoise/infer_01000000_denoise \
  --output_dir output/render-pipeline-denoise/picks \
  --pattern "*.obj,*.ply" \
  --resolution 512 \
  --samples 32
```

This writes one PNG per mesh plus `manifest.csv` and `index.html` in the output directory.

Create a denoising trajectory and render it as a video:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run --no-capture-output -n mflow python tools/save_denoising_trajectory.py \
  --config configs/snet/base-120m-ot-v-bench.yaml \
  --ckpt_path output/120m-ot-v-bench/checkpoints/01000000.pt \
  --output output/render-pipeline-denoise/infer_01000000_denoise \
  --num_steps 32

blender -b -P tools/render_video.py -- \
  --input output/render-pipeline-denoise/infer_01000000_denoise \
  --pattern "*.obj" \
  --output output/render-pipeline-denoise/render_denoising_white.mp4 \
  --frames_dir output/render-pipeline-denoise/render_denoising_white_frames \
  --transparent_frames \
  --background_color 1,1,1 \
  --denoise_hold_frames 3 \
  --hold_final_frames 8 \
  --rotate_frames 36
```

Use `--background_color 0,0,0` and omit `--transparent_frames` for a black-background render.

## Repository Structure

```text
configs/        experiment configs
datasets/       dataset loaders
models/         model components
tools/          launch, evaluation, and rendering scripts
utils/          logging, OT, and Chamfer utilities
flow_matching.py
inference.py
train.py
```

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
