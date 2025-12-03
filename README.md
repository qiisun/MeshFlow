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
conda env create -f env.yaml
conda activate mflow
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

cd ..
```
Then you should modify the `configs/vae.yaml`.

### Quick start
Train meshflow1
```bash
bash tools/run_train.sh configs/base.yaml
```

Train VAE
```
# train auto-encoder
bash tools/run_trainvae.sh configs/vae.yaml # regression loss
bash tools/run_trainvae.sh configs/vae_cls.yaml # classification loss


#eval auto-encoder
CUDA_VISIBLE_DEVICES=7, \
python eval_vae.py \
  --config configs/vae_cls.yaml \
  --checkpoint output/vae_cls/checkpoints/0060000.pt \
  --output_dir output/vae_cls/eval_samples \
  --num_save 20
```


Run JIT on dummy

```bash
# training
bash tools/run_train.sh configs/base_jit.yaml

CUDA_VISIBLE_DEVICES=7, python train_pixel_single.py --config configs/base_jit.yaml
```
