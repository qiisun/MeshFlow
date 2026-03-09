import os
import argparse
import yaml
import torch

from models.equidit import DiT
from transport import create_transport, Sampler
from datasets.mesh_dataset import save_mesh


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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
    parser.add_argument('--num-faces', type=int, default=800)
    parser.add_argument('--cfg-scale', type=float, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--max-val', type=float, default=2.653)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

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

    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg['sample'].get('cfg_scale', 1.0)
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
    with torch.inference_mode():
        while done < total:
            bs = min(args.batch_size, total - done)
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
                save_path = os.path.join(args.out_dir, f'{out_idx:06d}.obj')
                save_mesh(samples[i].detach().float().cpu().numpy(), save_path, max_val=args.max_val)

            done += bs
            print(f'[INFO] generated {done}/{total}')

    print(f'[DONE] Saved {total} meshes to: {args.out_dir}')


if __name__ == '__main__':
    main()
