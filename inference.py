import os
import argparse
import importlib
import importlib.util
import yaml
import torch
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import trimesh
# local imports
from flow_matching import create_transport
from datasets.mesh_dataset import ObjaverseDataset, save_mesh, float_to_index_np, index_to_float_np
from models.equidit import DiT

# Optional Chamfer distance extension
if importlib.util.find_spec("utils.chamfer3D.dist_chamfer_3D") is not None:
    chamfer_3DDist = importlib.import_module("utils.chamfer3D.dist_chamfer_3D").chamfer_3DDist
    CHAMFER_AVAILABLE = True
else:
    CHAMFER_AVAILABLE = False
    print("Warning: Chamfer distance not available")


def _configure_nccl_env_for_consumer_rtx():
    # Avoid Accelerate/NCCL initialization failure on RTX 40xx consumer GPUs.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


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
    
    points, _ = trimesh.sample.sample_surface(mesh, count=num_samples)
    return points


def _make_sample_fn(transport, sample_cfg, timestep_shift):
    transport.method = sample_cfg['sampling_method']
    transport.atol = sample_cfg['atol']
    transport.rtol = sample_cfg['rtol']
    num_steps = sample_cfg['num_sampling_steps']
    reverse = sample_cfg.get('reverse', False)
    t0, t1 = (1.0, 0.0) if reverse else (0.0, 1.0)

    def sample_fn(z, model_fn, **model_kwargs):
        wrapped_model = model_fn
        if timestep_shift > 0:
            def wrapped_model(x, t_scalar, **kwargs):
                shifted_t = transport.path.compute_time_shift(t_scalar, timestep_shift=timestep_shift)
                return model_fn(x, shifted_t, **kwargs)

        return transport.sample(
            z,
            wrapped_model,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            model_kwargs=model_kwargs,
        )

    return sample_fn


def _load_val_face_lengths(train_config):
    data_cfg = train_config.get('data', {})
    data_path = data_cfg.get('data_path', None)
    valid_path = data_cfg.get('valid_path', None)
    max_length = int(train_config['model'].get('max_length', 800))
    candidate_paths = []
    if valid_path:
        candidate_paths.append(valid_path)
    if data_path:
        candidate_paths.append(data_path)

    resolved_path = None
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, 'split', 'test.npz')):
            resolved_path = p
            break

    if resolved_path is None:
        return []

    val_dataset = ObjaverseDataset(
        data_pth=resolved_path,
        training=False,
        noise_sort=data_cfg.get('noise_sort', 'random'),
        use_custom_prior=False,
        do_dataset_normalize=True,
        use_rot_aug=False,
        use_scale_aug=False,
        use_permut_aug=False,
        max_face_length=max_length,
    )
    lengths = [int(item['faces_num']) for item in val_dataset.data if 0 < int(item['faces_num']) <= max_length]
    return lengths


# validation during training
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
        sample_cfg = train_config['sample']
        sample_fn = _make_sample_fn(transport, sample_cfg, timestep_shift)
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")
    
    want_chamfer = _should_compute_chamfer(train_config)
    compute_chamfer = want_chamfer and CHAMFER_AVAILABLE

    if want_chamfer and (not CHAMFER_AVAILABLE):
        print("Warning: Chamfer is requested but chamfer3D extension is unavailable, skipping Chamfer evaluation.")

    # Initialize Chamfer distance calculator only when needed
    if compute_chamfer:
        chamfer_dist = chamfer_3DDist()
    
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

                total_samples += 1
            # Use consistent max_val for saving mesh (1/0.3762)
            valid_tokens = cond_samples[b][valid_mask]
            save_mesh(valid_tokens.cpu().numpy(), f'{save_dir_mesh}/{i:03d}_{b:02d}.obj', max_val=1/0.3762)
    
    if was_training:
        model.train()

    # average validation loss on evaluated batches
    if val_steps == 0:
        return 0.0, 0.0, compute_chamfer
    
    avg_val_loss = running_val_loss / val_steps
    avg_chamfer_loss = running_chamfer_loss / total_samples if compute_chamfer and total_samples > 0 else 0.0
    
    return avg_val_loss, avg_chamfer_loss, compute_chamfer

# standalone inference for saving meshes (for generative metrics calculation, visualization, etc.)
@torch.no_grad()
def do_sample(train_config, accelerator, ckpt_path=None, cfg_scale=None, model=None, vae=None, demo_sample_mode=False):
    """Run mesh sampling and save .obj files."""
    del vae

    device = accelerator.device
    cfg_scale = train_config['sample'].get('cfg_scale', 1.0)
    timestep_shift = train_config['sample'].get('timestep_shift', 0.0)

    ckpt_path = train_config.get('ckpt_path', None)
    if ckpt_path is None:
        raise ValueError("ckpt_path must be provided either by argument or config['ckpt_path']")

    model = build_model_from_config(train_config)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['ema'])
    model = model.to(device)
    model.eval()

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        use_lognorm=train_config['transport'].get('use_lognorm', False),
        use_jit=train_config['transport'].get('use_jit', False),
    )
    sample_cfg = train_config['sample']
    sample_fn = _make_sample_fn(transport, sample_cfg, timestep_shift)

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
    # Prefer dedicated inference batch size and keep default conservative.
    batch_size = train_config['sample'].get(
        'infer_batch_size',
        train_config['sample'].get('per_proc_batch_size', 1)
    )
    if demo_sample_mode:
        batch_size = min(batch_size, 2)
    max_length = train_config['model'].get('max_length', 800)
    model_unwrap = accelerator.unwrap_model(model)

    val_face_lengths = _load_val_face_lengths(train_config)
    if len(val_face_lengths) == 0:
        raise ValueError(
            "Validation face lengths unavailable. Please ensure data.valid_path or data.data_path contains split/test.npz"
        )
    print(
        f"[INFO] Using validation face lengths: count={len(val_face_lengths)}, "
        f"min={min(val_face_lengths)}, max={max(val_face_lengths)}, "
        f"mean={float(np.mean(val_face_lengths)):.2f}"
    )
    length_ptr = 0

    saved = 0
    pbar = tqdm(total=num_samples, desc='Sampling meshes')
    while saved < num_samples:
        cur_bs = min(batch_size, num_samples - saved)
        z = torch.randn(cur_bs, max_length, 9, device=device)
        z_in = torch.cat([z, z], dim=0)

        idx = (np.arange(cur_bs) + length_ptr) % len(val_face_lengths)
        y_np = np.asarray(val_face_lengths, dtype=np.int64)[idx]
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
        mask = torch.arange(max_length, device=device).unsqueeze(0) < y.unsqueeze(1)
        length_ptr += cur_bs

        samples = sample_fn(z_in, model_unwrap.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)[-1]

        cond_samples = samples[:cur_bs]

        for b, item in enumerate(cond_samples):
            save_path = os.path.join(sample_folder_dir, f"{saved:06d}.obj")
            valid_len = int(y[b].item())
            valid_tokens = item[:valid_len]
            save_mesh(valid_tokens.detach().cpu().numpy(), save_path, max_val=1 / 0.3762)
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
    _configure_nccl_env_for_consumer_rtx()
    accelerator = Accelerator()
    train_config = load_config(args.config)

    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    ckpt_dir = train_config['ckpt_path']
    model = build_model_from_config(train_config)
    sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_dir, model=model, demo_sample_mode=args.demo)
    if accelerator.is_main_process:
        print(f"Saved samples to {sample_folder_dir}")
