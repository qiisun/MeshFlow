import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.mesh_dataset import ObjaverseDataset, collate_fn
from datasets.mesh_dataset import float_to_index_np, index_to_float_np
from models.equidit import DiT
from transport_simple import Sampler, create_transport

try:
    from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist

    CHAMFER3D_AVAILABLE = True
except Exception:
    chamfer_3DDist = None
    CHAMFER3D_AVAILABLE = False


def parse_steps(steps_text):
    steps = []
    for item in steps_text.split(','):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Sampling step must be positive, got: {value}")
        steps.append(value)
    if not steps:
        raise ValueError("No valid sampling steps are provided.")
    return steps


def load_checkpoint_config(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'config' not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'config'. Please pass --config explicitly from your training yaml."
        )
    config = checkpoint['config']
    return checkpoint, config


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


def build_train_loader_for_overfit(train_config, batch_size):
    dataset = ObjaverseDataset(
        data_pth=train_config['data']['data_path'],
        noise_sort=train_config['data']['noise_sort'],
        training=True,
        use_custom_prior=False,
        do_dataset_normalize=True,
        use_rot_aug=False,
        use_scale_aug=False,
        use_permut_aug=False,
        max_face_length=train_config['model'].get('max_length', 800),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: collate_fn(b, max_seq_length=train_config['model'].get('max_length', 800)),
    )
    return dataset, loader


def load_model_from_checkpoint(model, checkpoint):
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    return model


def chamfer_torch(pred_vertices, gt_vertices):
    pred = pred_vertices.unsqueeze(0)
    gt = gt_vertices.unsqueeze(0)
    pairwise = torch.cdist(pred, gt, p=2)
    dist_pred = pairwise.min(dim=2).values
    dist_gt = pairwise.min(dim=1).values
    return (dist_pred.mean() + dist_gt.mean()).item()


def tokens_to_mesh(tokens: np.ndarray, num_bins=2048, max_val=1.0 / 0.3762, std=0.3762):
    if tokens.ndim == 2:
        tokens = tokens.reshape(-1, 3, 3)

    tokens_rescaled = tokens * std
    vertices = tokens_rescaled.reshape(-1, 3).astype(np.float32)
    faces = np.arange(len(vertices)).reshape(-1, 3)

    vertices = float_to_index_np(vertices, min_val=-max_val, max_val=max_val, num_bins=num_bins)
    vertices = index_to_float_np(vertices, min_val=-max_val, max_val=max_val, num_bins=num_bins)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def sample_point_cloud(mesh: trimesh.Trimesh, num_samples: int = 4096):
    if mesh.vertices.shape[0] < 3:
        return np.tile(mesh.vertices[0], (num_samples, 1))

    try:
        points, _ = trimesh.sample.sample_surface(mesh, count=num_samples)
        return points
    except Exception:
        indices = np.random.choice(len(mesh.vertices), size=num_samples, replace=True)
        return mesh.vertices[indices]


@torch.no_grad()
def evaluate_step(
    model,
    valid_loader,
    device,
    transport,
    cfg_scale,
    timestep_shift,
    num_sampling_steps,
    max_batches,
    token_std,
    num_point_samples,
    print_range,
):
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method='euler',
        num_steps=num_sampling_steps,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=timestep_shift,
    )

    use_chamfer3d = CHAMFER3D_AVAILABLE
    chamfer3d = chamfer_3DDist() if use_chamfer3d else None

    total_chamfer = 0.0
    total_samples = 0
    has_printed_range = False

    pbar = tqdm(enumerate(valid_loader), total=min(len(valid_loader), max_batches), desc=f"Eval step={num_sampling_steps}")
    for batch_idx, data in pbar:
        if batch_idx >= max_batches:
            break

        x1 = data['tokens'].to(device)
        x0 = data['noise'].to(device)
        y = data['num_faces'].to(device)
        mask = data['masks'].to(device)

        z = torch.randn_like(x0, device=device)
        z = torch.cat([z, z], dim=0)

        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                samples = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)[-1]
        else:
            samples = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)[-1]

        bs = x0.shape[0]
        cond_samples = samples[:bs]
        for b in range(bs):
            valid_mask = mask[b].bool()
            pred_tokens = cond_samples[b][valid_mask].detach().cpu().numpy()
            gt_tokens = x1[b][valid_mask].detach().cpu().numpy()

            if pred_tokens.shape[0] == 0 or gt_tokens.shape[0] == 0:
                continue

            pred_mesh = tokens_to_mesh(pred_tokens, std=token_std)
            gt_mesh = tokens_to_mesh(gt_tokens, std=token_std)

            pred_points = sample_point_cloud(pred_mesh, num_samples=num_point_samples)
            gt_points = sample_point_cloud(gt_mesh, num_samples=num_point_samples)

            pred_vertices = torch.from_numpy(pred_points).unsqueeze(0).float().to(device)
            gt_vertices = torch.from_numpy(gt_points).unsqueeze(0).float().to(device)

            if print_range and (not has_printed_range):
                print(
                    "Range check | "
                    f"pred_token=[{pred_tokens.min():.4f}, {pred_tokens.max():.4f}] "
                    f"gt_token=[{gt_tokens.min():.4f}, {gt_tokens.max():.4f}] | "
                    f"pred_point=[{pred_points.min():.4f}, {pred_points.max():.4f}] "
                    f"gt_point=[{gt_points.min():.4f}, {gt_points.max():.4f}]"
                )
                has_printed_range = True

            if use_chamfer3d:
                dist1, dist2, _, _ = chamfer3d(
                    pred_vertices,
                    gt_vertices,
                )
                chamfer = (torch.mean(dist1) + torch.mean(dist2)).item()
            else:
                chamfer = chamfer_torch(pred_vertices.squeeze(0), gt_vertices.squeeze(0))

            total_chamfer += chamfer
            total_samples += 1

    if total_samples == 0:
        return float('nan')
    return total_chamfer / total_samples


def main():
    parser = argparse.ArgumentParser(description='Plot Chamfer distance vs inference sampling steps for one checkpoint.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--steps', type=str, default='5,10,20,30,50,75,100', help='Comma-separated sampling steps')
    parser.add_argument('--max-batches', type=int, default=0, help='How many batches to evaluate. 0 means full train set length.')
    parser.add_argument('--batch-size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--num-point-samples', type=int, default=4096, help='Number of uniformly sampled surface points per mesh')
    parser.add_argument('--out-dir', type=str, default='', help='Output directory for csv/json/png')
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    steps = parse_steps(args.steps)
    checkpoint, train_config = load_checkpoint_config(ckpt_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model_from_config(train_config).to(device)
    model = load_model_from_checkpoint(model, checkpoint)
    model.eval()

    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        use_lognorm=train_config['transport'].get('use_lognorm', False),
        use_jit=train_config['transport'].get('use_jit', False),
    )

    train_dataset, train_loader = build_train_loader_for_overfit(train_config, args.batch_size)

    cfg_scale = train_config.get('sample', {}).get('cfg_scale', 2.0)
    timestep_shift = train_config.get('sample', {}).get('timestep_shift', 0.0)
    token_std = train_config.get('data', {}).get('token_std', 0.3762)

    if not args.out_dir:
        exp_dir = os.path.dirname(os.path.dirname(ckpt_path))
        out_dir = os.path.join(exp_dir, 'analysis')
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.max_batches > 0:
        max_batches = args.max_batches
    else:
        max_batches = len(train_loader)

    print(f"GT split: train")
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval batches: {max_batches} / {len(train_loader)}")
    print(f"Chamfer backend: {'chamfer3D CUDA' if CHAMFER3D_AVAILABLE else 'torch.cdist fallback'}")
    print(f"Mesh surface sampling points per mesh: {args.num_point_samples}")
    print(f"Token std for mesh decode: {token_std}")

    results = []
    for num_steps in steps:
        chamfer = evaluate_step(
            model=model,
            valid_loader=train_loader,
            device=device,
            transport=transport,
            cfg_scale=cfg_scale,
            timestep_shift=timestep_shift,
            num_sampling_steps=num_steps,
            max_batches=max_batches,
            token_std=token_std,
            num_point_samples=args.num_point_samples,
            print_range=(len(results) == 0),
        )
        print(f"num_sampling_steps={num_steps:4d} -> chamfer={chamfer:.8f}")
        results.append({'num_sampling_steps': num_steps, 'chamfer': chamfer})

    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    csv_path = os.path.join(out_dir, f'chamfer_vs_steps_{ckpt_name}.csv')
    json_path = os.path.join(out_dir, f'chamfer_vs_steps_{ckpt_name}.json')
    fig_path = os.path.join(out_dir, f'chamfer_vs_steps_{ckpt_name}.png')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['num_sampling_steps', 'chamfer'])
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    xs = [item['num_sampling_steps'] for item in results]
    ys = [item['chamfer'] for item in results]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('Inference sampling steps')
    plt.ylabel('Chamfer distance')
    plt.title(f'Chamfer vs Inference Steps ({ckpt_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)

    best_idx = int(np.nanargmin(np.array(ys)))
    best_steps = xs[best_idx]
    best_chamfer = ys[best_idx]

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved FIG: {fig_path}")
    print(f"Best: steps={best_steps}, chamfer={best_chamfer:.8f}")


if __name__ == '__main__':
    main()