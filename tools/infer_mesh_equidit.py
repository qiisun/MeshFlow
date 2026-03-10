import os
import json
import argparse
import yaml
import torch
import trimesh
import numpy as np

import sys
sys.path.append(".")
from models.equidit import DiT
from models.equivae import float_to_index_np, index_to_float_np
from transport import create_transport, Sampler
from datasets.mesh_dataset import save_mesh
from tools.point_evaluation import sample_point_cloud, compute_all_metrics, jsd_between_point_cloud_sets


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_test_samples(data_path, max_samples=None, max_face_length=800):
    """Load ordered test samples from dataset files."""
    test_split_path = os.path.join(data_path, 'split', 'test.npz')
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Test split file not found: {test_split_path}")
    
    split = np.load(test_split_path, allow_pickle=True)['npz_list'].tolist()
    print(f"[INFO] Loaded {len(split)} test samples from split file")
    
    data_subfolder = "objaverse_occ_v5_ids"
    data_folder = os.path.join(data_path, data_subfolder)
    
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    
    samples = []
    for item in split:
        uid = item['uid']
        mesh_path = os.path.join(data_folder, f'{uid}.npz')
        if not os.path.exists(mesh_path):
            continue
        try:
            loaded_data = np.load(mesh_path, allow_pickle=True)
            num_faces = int(loaded_data['faces_num'])
            if 20 < num_faces < max_face_length:
                sample = {
                    'uid': uid,
                    'num_faces': num_faces,
                    'vertices': loaded_data['vertices'].astype(np.float32),
                    'faces': loaded_data['faces'].astype(np.int64),
                }
                samples.append(sample)
                if max_samples is not None and len(samples) >= max_samples:
                    break
        except Exception as e:
            print(f"[WARN] Error loading {mesh_path}: {e}")

    if len(samples) == 0:
        raise RuntimeError(f"No valid test samples found under {data_path}")

    face_counts = [item['num_faces'] for item in samples]
    print(f"[INFO] Loaded {len(samples)} valid test samples")
    print(f"[INFO] Face count range: [{min(face_counts)}, {max(face_counts)}]")
    print(f"[INFO] Mean face count: {np.mean(face_counts):.1f}")
    return samples


def tokens_to_mesh(tokens: np.ndarray, clean: bool = True, num_bins: int = 2048, max_val: float = 1.0):
    coords = tokens.reshape(-1, 3).astype(np.float32)
    vertices = float_to_index_np(coords, min_val=-max_val, max_val=max_val, num_bins=num_bins)
    vertices = index_to_float_np(vertices, min_val=-max_val, max_val=max_val, num_bins=num_bins)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if clean:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
    return mesh


def save_gt_mesh(vertices: np.ndarray, faces: np.ndarray, save_path: str):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(save_path)
    return mesh


def evaluate_generated_meshes(gen_meshes, gt_meshes, batch_size=32, num_points=2048, device=None):
    print(f"[INFO] Sampling {num_points} points per mesh for evaluation")
    gen_points = sample_point_cloud(gen_meshes, num_points=num_points)
    gt_points = sample_point_cloud(gt_meshes, num_points=num_points)

    jsd = float(jsd_between_point_cloud_sets(gen_points, gt_points))
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen_points_t = torch.from_numpy(gen_points).float().to(device)
    gt_points_t = torch.from_numpy(gt_points).float().to(device)
    metric_tensors = compute_all_metrics(gen_points_t, gt_points_t, batch_size=batch_size)

    metrics = {'JSD': jsd}
    for key, value in metric_tensors.items():
        if torch.is_tensor(value):
            metrics[key] = float(value.detach().cpu().item())
        else:
            metrics[key] = float(value)
    return metrics


def build_model(cfg):
    mc = cfg['model']
    model = DiT(
        hidden_dim=mc['hidden_dim'],
        num_heads=mc['num_heads'],
        max_length=mc['max_length'],
        num_layers=mc['num_layers'],
        gradient_checkpointing=mc.get('gradient_checkpointing', False),
        use_coord_encoding=mc.get('use_coord_encoding', True),
        version=mc.get('version', 3),
        pe_freq=mc.get('pe_freq', 20),
        mixed_precision=mc.get('mixed_precision', 'bf16'),
        use_dit_like_pe=mc.get('use_dit_like_pe', False),
        face_cond=mc.get('face_cond', True),
        face_bin=mc.get('face_bin', 20),
        use_rmsnorm=mc.get('use_rmsnorm', True),
        use_repa=cfg.get('train', {}).get('use_repa', False),
        is_latent=False,
    )
    return model


def load_checkpoint(model, ckpt_path, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if use_ema and 'ema' in ckpt:
            state_dict = ckpt['ema']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-faces', type=int, default=800, 
                        help='Default number of faces (used when --use-test-faces is False)')
    parser.add_argument('--use-test-faces', action='store_true', 
                        help='Use actual face counts from the validation/test split instead of fixed num-faces')
    parser.add_argument('--test-data-path', type=str, default=None,
                        help='Optional override for dataset root; defaults to config[data][data_path]')
    parser.add_argument('--cfg-scale', type=float, default=None,
                        help='Classifier-free guidance scale. Defaults to 1.0 if not specified.')
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--max-val', type=float, default=2.653)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--eval-num-points', type=int, default=2048)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    pred_dir = args.out_dir
    gt_dir = os.path.join(args.out_dir, 'gt_test_mesh')

    # Load ordered test samples if requested
    if args.use_test_faces:
        dataset_root = args.test_data_path or cfg['data']['data_path']
        print(f"[INFO] Loading validation/test samples from dataset: {dataset_root}")
        test_samples = load_test_samples(
            dataset_root,
            max_samples=args.num_samples,
            max_face_length=cfg['model'].get('max_length', args.num_faces),
        )
        if len(test_samples) < args.num_samples:
            print(f"[WARN] Only found {len(test_samples)} test samples, adjusting num_samples")
            args.num_samples = len(test_samples)
        os.makedirs(gt_dir, exist_ok=True)
    else:
        print(f"[INFO] Using fixed face count: {args.num_faces}")
        test_samples = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(device)
    model.eval()

    missing, unexpected = load_checkpoint(model, args.ckpt, use_ema=args.use_ema)
    if len(missing) > 0:
        print(f'[WARN] missing keys: {len(missing)}')
    if len(unexpected) > 0:
        print(f'[WARN] unexpected keys: {len(unexpected)}')

    transport = create_transport(
        cfg['transport']['path_type'],
        cfg['transport']['prediction'],
        cfg['transport']['loss_weight'],
        cfg['transport']['train_eps'],
        cfg['transport']['sample_eps'],
        use_cosine_loss=cfg['transport'].get('use_cosine_loss', False),
        use_lognorm=cfg['transport'].get('use_lognorm', False),
        use_jit=cfg['transport'].get('use_jit', False),
    )
    sampler = Sampler(transport)

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else 1.0
    num_steps = args.num_steps if args.num_steps is not None else cfg['sample']['num_sampling_steps']

    sample_fn = sampler.sample_ode(
        sampling_method=cfg['sample']['sampling_method'],
        num_steps=num_steps,
        atol=cfg['sample']['atol'],
        rtol=cfg['sample']['rtol'],
        reverse=cfg['sample'].get('reverse', False),
        timestep_shift=cfg['sample'].get('timestep_shift', 0.0),
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    total = args.num_samples
    done = 0
    generated_meshes = []
    gt_meshes = []
    with torch.inference_mode():
        while done < total:
            bs = min(args.batch_size, total - done)
            
            if test_samples is not None:
                batch_test_samples = test_samples[done:done+bs]
                batch_face_counts = [item['num_faces'] for item in batch_test_samples]
                max_faces_in_batch = max(batch_face_counts)
                
                z = torch.randn(bs, max_faces_in_batch, 9, device=device)
                y = torch.tensor(batch_face_counts, device=device, dtype=torch.long)
                
                mask = torch.zeros(bs, max_faces_in_batch, device=device, dtype=torch.bool)
                for b_idx, face_count in enumerate(batch_face_counts):
                    mask[b_idx, :face_count] = True
            else:
                # Use fixed face count
                z = torch.randn(bs, args.num_faces, 9, device=device)
                y = torch.full((bs,), args.num_faces, device=device, dtype=torch.long)
                mask = torch.ones(bs, args.num_faces, device=device, dtype=torch.bool)

            if cfg_scale > 1.0:
                z_in = torch.cat([z, z], dim=0)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    samples = sample_fn(z_in, model.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)[-1]
                samples = samples[:bs]
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    samples = sample_fn(z, model.forward, y=y, mask=mask)[-1]

            for i in range(bs):
                out_idx = done + i
                
                if test_samples is not None:
                    valid_faces = batch_face_counts[i]
                    sample_data = samples[i, :valid_faces].detach().float().cpu().numpy()
                else:
                    sample_data = samples[i].detach().float().cpu().numpy()
                
                save_path = os.path.join(pred_dir, f'{out_idx:06d}.obj')
                save_mesh(sample_data, save_path, max_val=args.max_val)
                generated_meshes.append(tokens_to_mesh(sample_data, max_val=args.max_val))

                if test_samples is not None:
                    gt_item = batch_test_samples[i]
                    gt_save_path = os.path.join(gt_dir, f'{out_idx:06d}_{gt_item["uid"]}.obj')
                    gt_mesh = save_gt_mesh(gt_item['vertices'], gt_item['faces'], gt_save_path)
                    gt_meshes.append(gt_mesh)

            done += bs
            
            if test_samples is not None:
                face_info = f" (faces: {batch_face_counts})"
            else:
                face_info = f" (faces: {args.num_faces})"
            print(f'[INFO] generated {done}/{total}{face_info}')

    print(f'[DONE] Saved {total} meshes to: {args.out_dir}')

    if test_samples is not None:
        print(f'[INFO] Saved GT test meshes to: {gt_dir}')
        print('[INFO] Computing point-based metrics...')
        metrics = evaluate_generated_meshes(
            generated_meshes,
            gt_meshes,
            batch_size=args.eval_batch_size,
            num_points=args.eval_num_points,
            device=device,
        )
        metrics_path = os.path.join(args.out_dir, 'point_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print('[INFO] Point evaluation results:')
        for key in sorted(metrics.keys()):
            print(f'  {key}: {metrics[key]:.8f}')
        print(f'[INFO] Metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()
