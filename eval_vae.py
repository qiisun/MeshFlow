import torch
import os
import argparse
import yaml
from tqdm import tqdm
from functools import partial
from accelerate import Accelerator

from models.equivae import loss_vae, AutoencoderKL, float_to_index, index_to_float
from datasets.mesh_dataset import ObjaverseDataset, collate_fn, save_mesh

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def evaluate(args):
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    config = load_config(args.config)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Init Model
    model = AutoencoderKL(latent_channels=config['model']['latent_channels'],
                          decoder_type=config['model']['decoder_type'],
                          num_bins=config['data']['num_bins'],
                          use_rmsnorm=config['model']['use_rms']).to(device)
    
    decoder_type = config['model']['decoder_type']
    num_bins = config['data']['num_bins']
    model.eval()

    # Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    if 'ema' in ckpt:
        print("Loading EMA weights...")
        model.load_state_dict(ckpt['ema'])
    else:
        print("Loading standard weights...")
        state_dict = ckpt['model']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    # Setup Dataset
    dataset = ObjaverseDataset(
        data_pth=config['data']['data_path'],
        training=False, 
        noise_sort=False,
        use_custom_prior=config['data'].get('use_custom_prior', False),
        use_decimated_dataset=config['data'].get('use_decimated_dataset', False),
        do_dataset_normalize=config['data'].get('do_dataset_normalize', False),
        vae=True,
        overfit=False,
        use_rot_aug=True,
        use_scale_aug=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=partial(collate_fn, max_seq_length=800)
    )

    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0
    num_batches = 0
    saved_count = 0
    total_acc = []

    print(f"Start Evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            x = data['tokens'].to(device)
            y = data['num_faces'].to(device)
            mask = data['masks'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Deterministic inference (Mode)
                recon, posterior, z = model(x, cond=y, mask=mask, sample_posterior=True)
                
                loss, rec_l, kl_l = loss_vae(
                    x, recon, posterior, mask=mask, 
                    kl_weight=config['train']['kl_weight'],
                    decoder_type=decoder_type, # 传入类型
                    num_bins=num_bins          # 传入 bins
                )

            total_loss += loss.item()
            total_rec += rec_l.item()
            total_kl += kl_l.item()
            num_batches += 1

            # Save Meshes
            if saved_count < args.num_save:
                bs = x.shape[0]
                
                if decoder_type == "cls":
                    # recon shape: [B, N, 9, num_bins] -> argmax -> [B, N, 9] (Indices)
                    target_indices = float_to_index(
                        x, 
                        min_val=-0.5, 
                        max_val=0.5, 
                        num_bins=num_bins
                    ).to(device)                    
                    recon_indices = torch.argmax(recon, dim=-1)
                    
                    correct = (recon_indices == target_indices)
                    mask_expanded = mask.unsqueeze(-1).expand_as(correct)
                    acc = correct[mask_expanded].float().mean()
                    total_acc.append(acc.item())

                    recon_coords = index_to_float(
                        recon_indices, 
                        min_val=-0.5,
                        max_val=0.5,
                        num_bins=num_bins
                    )
                else:
                    recon_coords = recon
                for b in range(bs):
                    if saved_count >= args.num_save: break
                    
                    mesh_np = recon_coords[b][mask[b]].float().cpu().numpy()
                    save_path = os.path.join(args.output_dir, f"eval_{saved_count:04d}.obj")
                    save_mesh(mesh_np, save_path)
                    saved_count += 1

    avg_loss = total_loss / num_batches
    avg_rec = total_rec / num_batches
    avg_kl = total_kl / num_batches
    avg_acc = sum(total_acc) / len(total_acc) if total_acc else 0  # <--- 计算平均准确率

    print("-" * 30)
    print(f"Avg Loss    : {avg_loss:.6f}")
    print(f"Avg Rec Loss: {avg_rec:.6f}")
    print(f"Avg KL Loss : {avg_kl:.6f}")
    print(f"Avg Accuracy: {avg_acc:.4%}")
    print(f"Saved {saved_count} meshes to {args.output_dir}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--num_save', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    evaluate(args)