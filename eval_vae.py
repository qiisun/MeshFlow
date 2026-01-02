import torch
import os
import argparse
import yaml
import numpy as np
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
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    sampling_dir = os.path.join(args.output_dir, "sampling_analysis")
    os.makedirs(sampling_dir, exist_ok=True)

    # Init Model
    model = AutoencoderKL(latent_channels=config['model']['latent_channels'],
                          decoder_type=config['model']['decoder_type'],
                          num_bins=config['data']['num_bins'],
                          use_rmsnorm=config['model']['use_rms'],
                          face_bin=config['model']['face_bin']).to(device)
    
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
        # 处理可能的 DDP 前缀
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
        use_rot_aug=False, # 评估时建议关掉旋转增强，方便对比
        use_scale_aug=False
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
    all_means = []
    all_stds = []

    # ==========================================
    # 定义采样时的噪声比重 (Temperature / Scale)
    # ==========================================
    # 0.0 = Deterministic (Mode), 1.0 = Standard Stochastic, >1.0 = High Variance
    noise_scales = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    # 只对前几个 batch 做详细采样分析，避免时间过长
    num_detailed_samples = 5 

    print(f"Start Evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            x = data['tokens'].to(device)
            bs, n_face, _ = x.shape
            y = data['num_faces'].to(device)
            mask = data['masks'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                recon, posterior, z = model(x, cond=y, mask=mask, sample_posterior=True)
                recon = x + torch.randn_like(x) * 0.02
                loss, rec_l, kl_l, mae = loss_vae(
                    x, recon, posterior, mask=mask, 
                    kl_weight=config['train']['kl_weight'],
                    decoder_type=decoder_type,
                    num_bins=num_bins
                )
            total_loss += loss.item()
            total_rec += rec_l.item()
            total_kl += kl_l.item()
            
            valid_means = posterior.mean.reshape(bs, n_face, -1)[mask]
            valid_stds = posterior.std.reshape(bs, n_face, -1)[mask]
            
            all_means.append(valid_means.float().cpu())
            all_stds.append(valid_stds.float().cpu())
            num_batches += 1
            
            if i < 0:
                idx = 0 
                curr_mean = posterior.mean[idx:idx+1] # [1, N, C]
                curr_std = posterior.std[idx:idx+1]   # [1, N, C]
                curr_mask = mask[idx:idx+1]
                curr_cond = y[idx:idx+1]
                
                for scale in noise_scales:
                    eps = torch.randn_like(curr_mean)
                    z_sampled = curr_mean + scale * curr_std * eps
                    
                    # Decode
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        recon_sampled = model.decode(z_sampled,cond=curr_cond, mask=curr_mask)
                    
                    # Convert to Mesh
                    mesh_np = process_recon_to_mesh(recon_sampled, curr_mask, decoder_type, num_bins)
                    
                    # Save: eval_{batch}_{type}_{scale}.obj
                    save_name = f"batch{i:02d}_sample{idx}_posterior_scale{scale:.1f}.obj"
                    save_mesh(mesh_np, os.path.join(sampling_dir, save_name))

                # --- B. Prior Sampling (无中生有，完全生成) ---
                z_prior = torch.randn_like(curr_mean) # Standard Normal
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    recon_prior = model.decode(z_prior,cond=curr_cond, mask=curr_mask)
                mesh_prior = process_recon_to_mesh(recon_prior, curr_mask, decoder_type, num_bins)
                save_mesh(mesh_prior, os.path.join(sampling_dir, f"batch{i:02d}_sample{idx}_prior_random.obj"))

            # =========================================================
            # 标准保存逻辑 (只保存 scale=1.0 的结果或 mean)
            # =========================================================
            if saved_count < args.num_save:
                bs = x.shape[0]
                # 转换整个 batch
                recon_coords_batch = convert_recon_output(recon, x, decoder_type, num_bins)
                
                for b in range(bs):
                    if saved_count >= args.num_save: break
                    
                    mesh_np = recon_coords_batch[b][mask[b]].float().cpu().numpy()
                    save_path = os.path.join(args.output_dir, f"eval_{saved_count:04d}_recon.obj")
                    save_mesh(mesh_np, save_path)
                    saved_count += 1
            
            # 简单计算 Acc
            if decoder_type == "cls":
                recon_indices = torch.argmax(recon, dim=-1)
                target_indices = float_to_index(x, -0.5, 0.5, num_bins).to(device)
                correct = (recon_indices == target_indices)
                mask_expanded = mask.unsqueeze(-1).expand_as(correct)
                acc = correct[mask_expanded].float().mean()
                total_acc.append(acc.item())

    avg_loss = total_loss / num_batches
    avg_rec = total_rec / num_batches
    avg_kl = total_kl / num_batches
    avg_acc = sum(total_acc) / len(total_acc) if total_acc else 0

    print("-" * 30)
    print(f"Avg Loss    : {avg_loss:.6f}")
    print(f"Avg Rec Loss: {avg_rec:.6f}")
    print(f"Avg KL Loss : {avg_kl:.6f}")
    print(f"Avg Accuracy: {avg_acc:.4%}")
    print(f"Detailed samples saved to: {sampling_dir}")
    print("-" * 30)
    
    
    all_means = torch.cat(all_means, dim=0) # [Total_All, C]
    all_stds = torch.cat(all_stds, dim=0)   # [Total_All, C]

    print("\n" + "="*40)
    print("       LATENT SPACE DIAGNOSIS       ")
    print("="*40)
    
    # 1. 均值 (Mean) 的分布情况
    # 理想情况下，全局均值应该接近 0
    global_mean_val = all_means.mean().item()
    global_mean_std = all_means.std().item() # 这是"均值的标准差"，代表 Latent 的活跃范围
    min_mean = all_means.min().item()
    max_mean = all_means.max().item()
    
    print(f"[Means (mu)]")
    print(f"  Avg Value : {global_mean_val:.6f} (Should be close to 0)")
    print(f"  Spread    : {global_mean_std:.6f} (Standard Deviation of the Means)")
    print(f"  Min / Max : {min_mean:.6f} / {max_mean:.6f}")
    
    # 2. 标准差 (Std) 的分布情况 (即模型的不确定性)
    # 理想情况下，这应该接近 1 (如果 KL 完美)，或者是一个较小的常数 (如果 KL 有权重)
    avg_sigma = all_stds.mean().item()
    min_sigma = all_stds.min().item()
    max_sigma = all_stds.max().item()
    
    print(f"\n[Stds (sigma)]")
    print(f"  Avg Value : {avg_sigma:.6f} (Average Uncertainty)")
    print(f"  Min / Max : {min_sigma:.6f} / {max_sigma:.6f}")

    # 3. 计算 Diffusion 需要的 Scaling Factor
    # Diffusion 希望输入的 Latent 是标准正态分布 (std approx 1)
    # Scaling Factor = 1 / Std(Means)
    rec_scale_factor = 1.0 / (global_mean_std + 1e-8)
    
    print(f"\n[Recommendation for Diffusion]")
    print(f"  Suggested Scaling Factor: {rec_scale_factor:.6f}")
    print(f"  (Use this to scale latents before feeding into Diffusion)")
    print("="*40 + "\n")

# =============================================
# Helper Functions (把转换逻辑提取出来，让主循环更干净)
# =============================================

def convert_recon_output(recon, x_target, decoder_type, num_bins):
    """根据 decoder 类型将输出转换为坐标"""
    if decoder_type == "cls":
        # Classification: Argmax -> Index -> Float
        recon_indices = torch.argmax(recon, dim=-1)
        recon_coords = index_to_float(
            recon_indices, 
            min_val=-0.5,
            max_val=0.5,
            num_bins=num_bins
        )
    else:
        recon_coords = recon
    return recon_coords

def process_recon_to_mesh(recon_tensor, mask, decoder_type, num_bins):
    """处理单个样本并返回 numpy 数组"""
    if decoder_type == "cls":
        recon_indices = torch.argmax(recon_tensor, dim=-1)
        coords = index_to_float(recon_indices, -0.5, 0.5, num_bins)
    else:
        coords = recon_tensor
        
    valid_coords = coords[0][mask[0]].float().cpu().numpy()
    return valid_coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--num_save', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    evaluate(args)