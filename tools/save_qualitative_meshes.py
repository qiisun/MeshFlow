import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets.mesh_dataset import ObjaverseDataset, collate_fn, save_mesh
from models.equidit import DiT
from transport import Sampler, create_transport


def parse_steps(steps_text):
    steps = []
    for item in steps_text.split(','):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Invalid step value: {value}")
        steps.append(value)
    if not steps:
        raise ValueError("No valid steps are provided")
    return steps


def load_checkpoint_and_config(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain config")
    return checkpoint, checkpoint['config']


def build_model_from_config(train_config):
    model_arch = train_config['model'].get('model_type', 'equidit')
    if model_arch != 'equidit':
        raise ValueError(f"Unsupported model_type: {model_arch}")

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


def load_model_weights(model, checkpoint):
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)


def build_train_loader(train_config, batch_size):
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
    return loader


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--steps', type=str, default='', help='Comma-separated steps, e.g. 5,10,20,50,100')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-save', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='')
    args = parser.parse_args()

    if args.steps:
        steps = parse_steps(args.steps)
    else:
        steps = [args.num_steps]

    checkpoint, train_config = load_checkpoint_and_config(args.ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model_from_config(train_config).to(device)
    load_model_weights(model, checkpoint)
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
    train_loader = build_train_loader(train_config, args.batch_size)
    data = next(iter(train_loader))

    x1 = data['tokens'].to(device)
    x0 = data['noise'].to(device)
    y = data['num_faces'].to(device)
    mask = data['masks'].to(device)

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    z_base = torch.randn_like(x0, device=device)
    z = torch.cat([z_base, z_base], dim=0)

    cfg_scale = train_config.get('sample', {}).get('cfg_scale', 1.0)
    max_val = 1.0 / 0.3762

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    if args.out_dir:
        out_dir = args.out_dir
    else:
        exp_dir = os.path.dirname(os.path.dirname(args.ckpt))
        if len(steps) == 1:
            out_dir = os.path.join(exp_dir, f'qualitative_{ckpt_name}_s{steps[0]}')
        else:
            out_dir = os.path.join(exp_dir, f'qualitative_{ckpt_name}_multi_steps')
    os.makedirs(out_dir, exist_ok=True)

    num_save = min(args.num_save, x0.shape[0])

    gt_dir = os.path.join(out_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    for i in range(num_save):
        valid_mask = mask[i].bool()
        gt_tokens = x1[i][valid_mask].detach().cpu().numpy()
        gt_path = os.path.join(gt_dir, f'{i:02d}_gt.obj')
        save_mesh(gt_tokens, gt_path, max_val=max_val)

    sampler = Sampler(transport)
    timestep_shift = train_config.get('sample', {}).get('timestep_shift', 0.0)
    for step in steps:
        sample_fn = sampler.sample_ode(
            sampling_method='euler',
            num_steps=step,
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            timestep_shift=timestep_shift,
        )

        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                samples = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)[-1]
        else:
            samples = sample_fn(z, model.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)[-1]

        cond_samples = samples[: x0.shape[0]]
        step_dir = os.path.join(out_dir, f'step_{step}')
        os.makedirs(step_dir, exist_ok=True)

        for i in range(num_save):
            valid_mask = mask[i].bool()
            pred_tokens = cond_samples[i][valid_mask].detach().cpu().numpy()
            pred_path = os.path.join(step_dir, f'{i:02d}_pred.obj')
            save_mesh(pred_tokens, pred_path, max_val=max_val)

    print(f'Saved qualitative meshes to: {out_dir}')
    print(f'Saved pairs per step: {num_save}')
    print(f'Steps: {steps}')


if __name__ == '__main__':
    main()