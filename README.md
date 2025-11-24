# MeshFlow

MeshFlow based on lightingDiT.


### TODO
- [x] clean code
- [x] simple train & test
- [x] implement jit
- [x] DDP for single-node multi-card training  (test error in evaluation, fixed)
- [ ] dynamic allocator

```bash
bash run_train.sh configs/base.yaml

bash run_train.sh configs/base_jit.yaml
```