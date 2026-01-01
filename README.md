# MeshFlow

MeshFlow based on lightingDiT.


### TODO
- [x] clean code
- [x] simple train & test
- [x] implement jit
- [x] DDP for single-node multi-card training  (test error in evaluation, fixed)
- [ ] dynamic allocator
- [x] prepare shapenet dataset (full)
- [x] prepare objaverse dataset


### Environment

```bash
conda create -n mflow python=3.10 -y
conda activate mflow

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# flash attention
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -r requirements.txt

# partfield
mkdir third_party
cd third_party
git clone https://github.com/nv-tlabs/PartField.git
cd PartField
mkdir model
wget https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt
cd ..
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
apt install libx11-6 libgl1 libxrender1
pip install vtk

# p3sam
cd third_party/Hunyuan3D-Part/P3-SAM

pip install viser fpsample trimesh numba gradio
CUDA_VISIBLE_DEVICES=6, python auto_mask.py --mesh_path assets --output_path results/all

```


### dataset

```bash
mkdir downloaded_data
cd downloaded_data
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/dummy.tar.gz # or objaverse
tar xf dummy.tar.gz
rm dummy.tar.gz


wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet.tar.gz # or objaverse
tar xf shapenet.tar.gz
rm shapenet.tar.gz


wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/objaverse_occ_v5_ids.tar.gz 
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/split.tar.gz
tar xf objaverse_occ_v5_ids.tar.gz
rm objaverse_occ_v5_ids.tar.gz
tar xf split.tar.gz
rm split.tar.gz
mkdir objaverse
mv objaverse_occ_v5_ids objaverse
mv split objaverse


wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet-cls.tar.gz
tar xf shapenet-cls.tar.gz
rm shapenet-cls.tar.gz

cd ..
```
Then you should modify the `configs/vae.yaml`.

### Train MeshFlow-1
```bash
bash tools/run_train.sh configs/base.yaml
```

### Train Latent MeshFlow
```bash
# change the 
bash tools/run_train_latent.sh configs/latent.yaml

CUDA_VISIBLE_DEVICES=6, \
accelerate launch eval_ldm.py \
  --config configs/latent.yaml \
  --ckpt output/ldm/checkpoints/0070000.pt \
  --out_dir output/ldm \
  --use_ema \
  --batch_size 4
```

### Train VAE & Evaluate VAE
```
# train auto-encoder
bash tools/run_trainvae.sh configs/vae.yaml # regression loss
bash tools/run_trainvae.sh configs/vae_cls.yaml # classification loss


# eval auto-encoder (L1 loss)
mkdir -p output/vae_rms_lamp/checkpoints
cd output/vae_rms_lamp/checkpoints
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/vae_ckpts_1e-3/0036000.pt
# wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/vae_ckpts/0130000.pt
cd ../../..

CUDA_VISIBLE_DEVICES=0, \
python eval_vae.py \
  --config configs/vae_ch4_1e-2.yaml \
  --checkpoint output/vae_rms_lamp_ch4/checkpoints/00000.pt \
  --output_dir output/vae_rms_lamp_ch4/eval_samples \
  --num_save 40
```


### Run JIT on dummy

```bash
# training
bash tools/run_train.sh configs/base_jit.yaml

CUDA_VISIBLE_DEVICES=6, python train_pixel_single.py --config configs/base_jit.yaml
```


### Extract feature

```bash
cd third_party/PartField
CUDA_VISIBLE_DEVICES=6, python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/objaverse dataset.data_path ../../downloaded_data/dummy/objaverse_occ_v5_ids
# 注意如果有文件的话，会跳过

cd ../..
python tools/merge_feature.py
bash tools/run_train_421a.sh configs/base.yaml
```