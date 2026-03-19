"""
Sampling Scripts of LightningDiT.

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import trimesh
# local imports
from transport import create_transport, Sampler
from datasets.mesh_dataset import save_mesh, float_to_index_np, index_to_float_np
from models.equidit import DiT

# Try to import Chamfer distance
try:
    from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
    CHAMFER_AVAILABLE = True
except:
    CHAMFER_AVAILABLE = False
    print("Warning: Chamfer distance not available")


def _should_compute_chamfer(train_config):
    sample_cfg = train_config.get('sample', {})
    if 'compute_chamfer' in sample_cfg:
        return bool(sample_cfg['compute_chamfer'])

    data_cfg = train_config.get('data', {})
    train_cfg = train_config.get('train', {})
    data_path = str(data_cfg.get('data_path', '')).lower()
    exp_name = str(train_cfg.get('exp_name', '')).lower()

    return bool(
        data_cfg.get('overfit', False)
        or 'overfit' in data_path
        or 'ss_overfit' in data_path
        or 'overfit' in exp_name
    )

def tokens_to_mesh(tokens: np.ndarray, num_bins=204800, max_val=1, std=0.3762):
    """
    Convert tokens to trimesh object.
    tokens: [N, 3, 3] or [N, 9] triangle soup format
    
    Note: tokens are in scaled space (divided by std).
    We need to rescale them back before creating mesh.
    """
    # Reshape if needed
    if tokens.ndim == 2:
        tokens = tokens.reshape(-1, 3, 3)
    
    # Rescale tokens back to original scale (multiply by std)
    tokens_rescaled = tokens * std
    
    # Extract vertices and faces
    vertices = tokens_rescaled.reshape(-1, 3).astype(np.float32)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    
    # Dequantize
    vertices = float_to_index_np(vertices, min_val=-max_val, max_val=max_val, num_bins=num_bins)
    vertices = index_to_float_np(vertices, min_val=-max_val, max_val=max_val, num_bins=num_bins)
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh

def sample_point_cloud(mesh: trimesh.Trimesh, num_samples: int = 4096):
    """
    Sample points from mesh surface.
    Returns: [num_samples, 3] point cloud
    """
    if mesh.vertices.shape[0] < 3:
        # If mesh is too small, just return vertices repeated
        return np.tile(mesh.vertices[0], (num_samples, 1))
    
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count=num_samples)
        return points
    except:
        # Fallback: sample from vertices randomly
        indices = np.random.choice(len(mesh.vertices), size=num_samples, replace=True)
        return mesh.vertices[indices]

@torch.no_grad()
def do_sample_simple(
    model,
    valid_loader,
    device,
    transport,
    train_config,
    accelerator,
    train_steps,
    save_dir,
    cfg_scale=None,
    timestep_shift=None,
):
    sampler = Sampler(transport)
    mode = train_config['sample'].get('mode', 'ODE')
    if timestep_shift is None:
        timestep_shift = train_config['sample'].get('timestep_shift', 0.0)
    if cfg_scale is None:
        cfg_scale = train_config['sample'].get('cfg_scale', 1.0)
    was_training = model.training
    model.eval()
    
    save_dir_mesh = f'{save_dir}/mesh_{train_steps}step'
    os.makedirs(save_dir_mesh, exist_ok=True)

    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")
    
    compute_chamfer = _should_compute_chamfer(train_config) and CHAMFER_AVAILABLE

    if _should_compute_chamfer(train_config) and (not CHAMFER_AVAILABLE):
        print("Warning: Chamfer is requested but chamfer3D extension is unavailable, skipping Chamfer evaluation.")

    # Initialize Chamfer distance calculator only when needed
    if compute_chamfer:
        chamfer_dist = chamfer_3DDist()
    
    images = []
    running_val_loss = 0.0
    running_chamfer_loss = 0.0
    val_steps = 0    
    total_samples = 0    
    for i, data in tqdm(enumerate(valid_loader), desc="Generating Demo Samples"):
        if i > 10:
            break
        # supports batch_size >= 1
        x1 = data['tokens'].to(device)
        x0 = data['noise'].to(device) # presampled noise
        y = data['num_faces'].to(device)
        mask = data['masks'].to(device)

        model_kwargs = dict(y=y, mask=mask)
        # Use autocast during validation to match training behavior and ensure consistent dtype
        # When accelerator is configured with mixed_precision, we need explicit autocast during forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict = transport.training_losses(model, x1, x0, model_kwargs)
        cur_loss = loss_dict["loss"].mean()
        running_val_loss += cur_loss.item()
        val_steps += 1

        z = torch.randn_like(x0, device=device) # random sample noise
        z = torch.cat([z, z], 0)
        model_unwrap = accelerator.unwrap_model(model)
        # forward_with_cfg internally builds unconditional labels using model_unwrap.uncond_y
        # so we only pass conditional y here.
        model_kwargs = dict(y=y, cfg_scale=cfg_scale, mask=mask)
        model_fn = model_unwrap.forward_with_cfg
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # z: [2*bs, N, 9]
            # y: [bs]
            # mask: [bs, N], no repeat
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        images.append(samples)
        # save meshes (only conditional half)
        bs = x0.shape[0]
        cond_samples = samples[:bs]
        
        # Extract valid tokens using mask
        for b in range(bs):
            valid_mask = mask[b].bool()
            pred_vertices = cond_samples[b][valid_mask][:, :3]  # [N_valid, 3]
            gt_vertices = x1[b][valid_mask][:, :3]  # [N_valid, 3]

            # Chamfer distance using mesh-based point sampling
            if compute_chamfer:
                try:
                    # Convert tokens to mesh and sample point clouds
                    pred_tokens = cond_samples[b][valid_mask].cpu().numpy()  # [N_valid, feature_dim]
                    gt_tokens = x1[b][valid_mask].cpu().numpy()  # [N_valid, feature_dim]
                    
                    # Use std=0.3762 (from dataset) and max_val=1/0.3762 (dequantization range)
                    MAX_VAL = 1.0 / 0.3762  # = 2.653
                    STD = 0.3762
                    
                    pred_mesh = tokens_to_mesh(pred_tokens, num_bins=2048, max_val=MAX_VAL, std=STD)
                    gt_mesh = tokens_to_mesh(gt_tokens, num_bins=2048, max_val=MAX_VAL, std=STD)
                    
                    # Sample 4096 points from each mesh
                    pred_pc = sample_point_cloud(pred_mesh, num_samples=4096)  # [4096, 3]
                    gt_pc = sample_point_cloud(gt_mesh, num_samples=4096)  # [4096, 3]
                    
                    # Convert to torch tensors
                    pred_pc_torch = torch.from_numpy(pred_pc).unsqueeze(0).float().to(device)  # [1, 4096, 3]
                    gt_pc_torch = torch.from_numpy(gt_pc).unsqueeze(0).float().to(device)  # [1, 4096, 3]
                    
                    # Calculate Chamfer distance
                    dist1, dist2, _, _ = chamfer_dist(pred_pc_torch, gt_pc_torch)
                    chamfer = (torch.mean(dist1) + torch.mean(dist2)).item()
                    running_chamfer_loss += chamfer
                except Exception as e:
                    # If Chamfer calculation fails, use vertex-based distance
                    pred_vertices_torch = pred_vertices.unsqueeze(0)  # [1, N_valid, 3]
                    gt_vertices_torch = gt_vertices.unsqueeze(0)  # [1, N_valid, 3]
                    dist1, dist2, _, _ = chamfer_dist(pred_vertices_torch, gt_vertices_torch)
                    chamfer = (torch.mean(dist1) + torch.mean(dist2)).item()
                    running_chamfer_loss += chamfer

                total_samples += 1
            # Use consistent max_val for saving mesh (1/0.3762)
            save_mesh(cond_samples[b].cpu().numpy(), f'{save_dir_mesh}/{i:03d}_{b:02d}.obj', max_val=1/0.3762)
    
    if was_training:
        model.train()

    # average validation loss on evaluated batches
    if val_steps == 0:
        return 0.0, 0.0, compute_chamfer
    
    avg_val_loss = running_val_loss / val_steps
    avg_chamfer_loss = running_chamfer_loss / total_samples if compute_chamfer and total_samples > 0 else 0.0
    
    return avg_val_loss, avg_chamfer_loss, compute_chamfer


def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None, demo_sample_mode=False):
    """Run mesh sampling and save .obj files."""
    del vae

    device = accelerator.device
    if cfg_scale is None:
        cfg_scale = train_config['sample'].get('cfg_scale', 1.0)
    timestep_shift = train_config['sample'].get('timestep_shift', 0.0)

    if ckpt_path is None:
        ckpt_path = train_config.get('ckpt_path', None)
    if ckpt_path is None:
        raise ValueError("ckpt_path must be provided either by argument or config['ckpt_path']")

    if model is None:
        model = build_model_from_config(train_config)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss=train_config['transport'].get('use_cosine_loss', False),
        use_lognorm=train_config['transport'].get('use_lognorm', False),
        use_jit=train_config['transport'].get('use_jit', False),
    )
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=train_config['sample']['sampling_method'],
        num_steps=train_config['sample']['num_sampling_steps'],
        atol=train_config['sample']['atol'],
        rtol=train_config['sample']['rtol'],
        reverse=train_config['sample']['reverse'],
        timestep_shift=timestep_shift,
    )

    step_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    sample_folder_dir = os.path.join(
        train_config['train']['output_dir'],
        train_config['train']['exp_name'],
        f"infer_{step_name}",
    )
    os.makedirs(sample_folder_dir, exist_ok=True)

    num_samples = train_config['sample'].get('num_samples', 100)
    if demo_sample_mode:
        num_samples = min(num_samples, 8)
    batch_size = train_config['sample'].get('per_proc_batch_size', 4)
    max_length = train_config['model'].get('max_length', 800)
    model_unwrap = accelerator.unwrap_model(model)

    saved = 0
    pbar = tqdm(total=num_samples, desc='Sampling meshes')
    while saved < num_samples:
        cur_bs = min(batch_size, num_samples - saved)
        z = torch.randn(cur_bs, max_length, 9, device=device)
        z_in = torch.cat([z, z], dim=0)
        y = torch.full((cur_bs,), max_length, device=device, dtype=torch.long)
        mask = torch.ones((cur_bs, max_length), device=device, dtype=torch.bool)
        samples = sample_fn(z_in, model_unwrap.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)[-1]
        cond_samples = samples[:cur_bs]

        for item in cond_samples:
            save_path = os.path.join(sample_folder_dir, f"{saved:06d}.obj")
            save_mesh(item.detach().cpu().numpy(), save_path, max_val=1 / 0.3762)
            saved += 1
            pbar.update(1)

    pbar.close()
    return sample_folder_dir

def build_model_from_config(train_config):
    model_arch = train_config['model'].get('model_type', 'equidit')
    if model_arch != 'equidit':
        raise ValueError(f"Unsupported model_type: {model_arch}. Only 'equidit' is supported.")

    return DiT(
        hidden_dim=train_config['model']['hidden_dim'],
        num_heads=train_config['model']['num_heads'],
        max_length=train_config['model']['max_length'],
        num_layers=train_config['model']['num_layers'],
        gradient_checkpointing=train_config['model']['gradient_checkpointing'],
        use_coord_encoding=train_config['model']['use_coord_encoding'],
        version=train_config['model']['version'],
        pe_freq=train_config['model']['pe_freq'],
        mixed_precision=train_config['model']['mixed_precision'],
        use_dit_like_pe=train_config['model']['use_dit_like_pe'],
        face_cond=train_config['model']['face_cond'],
        face_bin=train_config['model']['face_bin'],
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
    )

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_jit.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    ckpt_dir = train_config['ckpt_path']
    model = build_model_from_config(train_config)
    sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_dir, model=model, demo_sample_mode=args.demo)
    if accelerator.is_main_process:
        print(f"Saved samples to {sample_folder_dir}")
