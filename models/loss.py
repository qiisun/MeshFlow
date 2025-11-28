import torch
import torch.nn.functional as F

def mesh_loss(x1, x0, model, model_kwargs):
    output = model(x0, t=0.95, **model_kwargs)*0.05+x0 # xt+ (1-t)*v_t
    
    loss = F.l1_loss(x1, output)
    return loss