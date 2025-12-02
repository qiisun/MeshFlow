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

### dataset

```bash
mkdir downloaded_data & cd downloaded_data
wget https://huggingface.co/datasets/qsun2001/omg/resolve/main/obj_data/shapenet.tar.gz # or objaverse
tar xf shapenet.tar.gz
rm shapenet.tar.gz
cd ..
```
Then you should modify the `configs/vae.yaml`.

### Quick start
```bash
bash tools/run_train.sh configs/base.yaml

# donot use jit
bash tools/run_train.sh configs/base_jit.yaml


# train auto-encoder
bash tools/run_trainvae.sh configs/vae.yaml # regression loss
bash tools/run_trainvae.sh configs/vae_cls.yaml # classification loss


#eval auto-encoder
CUDA_VISIBLE_DEVICES=2, \
python eval_vae.py \
  --config configs/vae_cls.yaml \
  --checkpoint output/vae_cls/checkpoints/0040000.pt \
  --output_dir output/vae_cls/eval_samples \
  --num_save 20
```