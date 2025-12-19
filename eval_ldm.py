"""
Evaluation Script for Latent DiT (LDM) for 3D Meshes.
Loads a checkpoint and performs sampling/validation.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml
import os
import argparse
from copy import deepcopy
from collections import OrderedDict
from functools import partial
import logging
from tqdm import tqdm

from models.equivae import AutoencoderKL
from transport import create_transport
from accelerate import Accelerator
from datasets.mesh_dataset import ObjaverseDataset, collate_fn
from inference import do_sample_simple, save_mesh # 确保 inference.py 里有 save_mesh

from models.dit import DiT_Llama_600M_patch1 

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO, 
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/eval_log.txt")]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the training config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--out_dir', type=str, default='eval_results', help='Directory to save results')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA weights for evaluation (Recommended)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of meshes to generate')
    args = parser.parse_args()

    # 1. Setup Accelerator & Logger
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(args.out_dir, exist_ok=True)
    logger = create_logger(args.out_dir)
    
    config = load_config(args.config)
    
    if accelerator.is_main_process:
        logger.info(f"Loading Checkpoint: {args.ckpt}")
        logger.info(f"Using EMA: {args.use_ema}")

    # 2. VAE Setup (用于解码 Latents)
    vae_config = config['vae']
    vae = AutoencoderKL(
        latent_channels=vae_config['latent_channels'], 
        decoder_type=vae_config['decoder_type'],
        num_bins=config['data']['num_bins'],
        use_rmsnorm=vae_config.get('use_rmsnorm', False)
    ).to(device)

    if 'vae_ckpt' in vae_config:
        vae_ckpt_path = vae_config['vae_ckpt']
        if os.path.exists(vae_ckpt_path):
            vae_ckpt = torch.load(vae_ckpt_path, map_location='cpu')
            if 'ema' in vae_ckpt:
                vae.load_state_dict(vae_ckpt['ema'])
            else:
                vae.load_state_dict(vae_ckpt['model'])
            if accelerator.is_main_process:
                logger.info(f"Loaded VAE from {vae_ckpt_path}")
        else:
            logger.warning(f"VAE checkpoint not found at {vae_ckpt_path}, using random init (WARNING!)")
    
    vae.eval()
    latent_scale_factor = vae_config.get('scale_factor', 1.0) # 重要参数

    # 3. Model Setup (DiT)
    # 请确保这里的模型定义和训练时完全一致
    model = DiT_Llama_600M_patch1()
    # 如果使用配置创建模型:
    # model = DiT(
    #     hidden_dim=config['model']['hidden_dim'], 
    #     ...
    # )
    model = model.to(device)

    # 4. Load Checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    
    if args.use_ema and 'ema' in checkpoint:
        logger.info("Loading EMA weights...")
        model.load_state_dict(checkpoint['ema'])
    else:
        logger.info("Loading Standard weights...")
        model.load_state_dict(checkpoint['model'])
    
    model.eval()

    # 5. Transport Setup (用于采样)
    transport = create_transport(
        config['transport']['path_type'],
        config['transport']['prediction'],
        config['transport']['loss_weight'],
        config['transport']['train_eps'],
        config['transport']['sample_eps'],
        use_cosine_loss=config['transport'].get('use_cosine_loss', False),
        use_lognorm=config['transport'].get('use_lognorm', False),
        use_jit=config['transport'].get('use_jit', False),
    )

    # 6. Data Setup (Validation Set)
    if 'valid_path' in config['data']:
        valid_dataset = ObjaverseDataset(
            data_pth=config['data']['data_path'], # 或者 config['data']['valid_path'] 如果分开存放
            training=False,
            noise_sort=config['data']['noise_sort'],
            use_decimated_dataset=False,
            do_dataset_normalize=False,
            use_rot_aug=False,
            use_scale_aug=False,
            use_repa=False,
            use_permut_aug=False
        )
        
        # 限制验证集大小，避免太慢
        # valid_dataset = torch.utils.data.Subset(valid_dataset, range(min(len(valid_dataset), args.num_samples)))

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False, # 固定顺序
            num_workers=4,
            drop_last=False,
            collate_fn=partial(collate_fn, max_seq_length=800)
        )
    else:
        logger.error("No valid_path found in config!")
        return

    # 7. Run Evaluation
    logger.info("Starting Evaluation...")
    step_num = int(args.ckpt.split('/')[-1].split('.')[0]) if args.ckpt.split('/')[-1][0].isdigit() else 0
        
    val_loss = do_sample_simple(
        model=model,
        valid_loader=valid_loader,
        device=device,
        transport=transport,
        train_config=config,
        accelerator=accelerator,
        train_steps=step_num,
        save_dir=args.out_dir,
        vae=vae,
        latent_scale_factor=latent_scale_factor,
        cfg_scale = 10
    )

    logger.info(f"Evaluation Finished.")
    logger.info(f"Avg Validation Loss: {val_loss:.4f}")
    logger.info(f"Results saved to: {args.out_dir}/mesh_{step_num}step")

if __name__ == "__main__":
    main()