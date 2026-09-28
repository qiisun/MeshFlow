# MeshFlow: Single-GPU Quick Start

Generate meshes from a pretrained model on a clean Ubuntu installation.
No training is required.

**Requirements:** Ubuntu x86_64, one NVIDIA Ampere/Ada/Hopper GPU with
BF16 support (e.g. RTX 3090, RTX 4090, or A100), and an NVIDIA driver
compatible with CUDA 12.4 (550.54.15 or newer). A 24 GB GPU is recommended,
not a measured minimum. Allow approximately 20 GB of free disk space.

**Note:** these instructions have not yet been tested end-to-end on
Ubuntu 26.04.

## 1. Prepare Ubuntu

```bash
sudo apt update
sudo apt install -y git curl ca-certificates build-essential ubuntu-drivers-common
nvidia-smi
```

If the NVIDIA driver is not installed, run the following and reboot.
Complete any Secure Boot enrollment requested by Ubuntu.

```bash
sudo ubuntu-drivers install
sudo reboot
```

Continue only when `nvidia-smi` lists your GPU.

## 2. Install Conda and clone MeshFlow

If Conda is already installed, skip the Miniforge installer and use your
existing Conda installation. Otherwise:

```bash
curl -fL \
  https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh \
  -o /tmp/meshflow-miniforge.sh
bash /tmp/meshflow-miniforge.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Create the environment and download the code:

```bash
conda create -n mflow --override-channels -c conda-forge python=3.10 pip -y
conda activate mflow
git clone https://github.com/qiisun/MeshFlow.git
cd MeshFlow
```

Run all remaining commands from this directory in the same terminal.

## 3. Install dependencies

The current code builds a Chamfer CUDA extension during import, even for
inference. Install CUDA development tools and GCC 12 inside Conda to avoid
depending on Ubuntu's system compiler:

```bash
conda install --override-channels \
  -c nvidia/label/cuda-12.4.1 -c conda-forge \
  cuda-toolkit=12.4.1 gcc_linux-64=12 gxx_linux-64=12 \
  sysroot_linux-64=2.28 -y
conda deactivate
conda activate mflow

export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=2
export CUDA_VISIBLE_DEVICES=0
```

Install the focused inference/evaluation dependencies below instead of
the broader historical `requirements.txt` export:

```bash
python -m pip install pip==25.0.1 setuptools==75.6.0 wheel==0.45.1
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  numpy==1.26.4 scipy==1.15.3 trimesh==4.10.0 \
  einops==0.8.1 accelerate==1.12.0 PyYAML==6.0.3 \
  tqdm==4.67.1 torchdiffeq==0.2.5 scikit-learn==1.7.2 \
  ninja==1.13.0 packaging==25.0
python -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -m pip install --no-build-isolation --no-deps ./utils/chamfer3D
python -m pip check
```

Use these Python/PyTorch versions with the specified FlashAttention wheel.

## 4. Download data and the pretrained model

Inference needs the test data to obtain mesh face counts. Download the
category-split archive and the bench checkpoint:

```bash
mkdir -p downloaded_data output/120m-ot-v-bench/checkpoints

curl -fL --retry 3 \
  https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/obj_data/shapenet-cls.tar.gz \
  -o downloaded_data/shapenet-cls.tar.gz
tar -xzf downloaded_data/shapenet-cls.tar.gz -C downloaded_data

curl -fL --retry 3 \
  https://huggingface.co/datasets/qsun2001/meshflow/resolve/main/v1/120m-ot-v-bench/checkpoints/last.pt \
  -o output/120m-ot-v-bench/checkpoints/last.pt
```

## 5. Generate meshes

Generate eight bench meshes using one GPU and batch size 1:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --config configs/snet/base-120m-ot-v-bench.yaml \
  ckpt_path=output/120m-ot-v-bench/checkpoints/last.pt \
  data.data_path=downloaded_data/shapenet-02828884 \
  data.valid_path=downloaded_data/shapenet-02828884 \
  train.exp_name=single-gpu-bench \
  sample.num_samples=8 \
  sample.infer_batch_size=1
```

Expected output: eight OBJ files, `000000.obj` through `000007.obj`, in
`output/single-gpu-bench/infer_last/`.

## 6. Evaluate

For a quick check of the evaluation pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/point_evaluation.py \
  --gen-root output/single-gpu-bench/infer_last \
  --data-root downloaded_data \
  --category bench \
  --max-gen-meshes 8 \
  --batch-size 1 \
  --num-runs 1
```

Results are saved to `point_eval_summary.json` in the generated mesh
directory. It should contain a nonempty `02828884` result.

**For quantitative evaluation:** rerun generation with
`sample.num_samples=1000`, then evaluate without `--max-gen-meshes 8`
and with `--num-runs 5`. The evaluator defaults to the valid test-set
size. Report CD-based metrics and JSD, not the placeholder EMD values.
Eight-mesh smoke-test results should not be compared with the paper.

If GPU memory runs out, keep both batch sizes at 1 and use a GPU with more
available memory. When reporting results or errors, include your GPU,
driver, Ubuntu version, exact command, and Git revision (`git rev-parse HEAD`).
