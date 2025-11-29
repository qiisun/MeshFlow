# MeshFlow

MeshFlow based on lightingDiT.


### TODO
- [x] clean code
- [x] simple train & test
- [x] implement jit
- [x] DDP for single-node multi-card training  (test error in evaluation, fixed)
- [ ] dynamic allocator
- [ ] prepare shapenet dataset (lamp)

```bash
bash tools/run_train.sh configs/base.yaml

# donot use jit
bash tools/run_train.sh configs/base_jit.yaml


# train auto-encoder
bash tools/run_trainvae.sh configs/vae.yaml
```


