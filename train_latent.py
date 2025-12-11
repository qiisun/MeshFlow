"""
Training Codes of Latent DiT (LDM) for 3D Meshes.
Trains a DiT to generate VAE latents instead of raw coordinates.

by Maple (Jingfeng Yao) from HUST-VL
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils.logger import WandBLogger as SummaryWriter

import math
import yaml
import json
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict

# --- Models ---
from models.equidit import DiT
from models.equivae import AutoencoderKL # Import VAE
from transport import create_transport
from accelerate import Accelerator
from inference import do_sample_simple # You might need to update inference to decode latents

from datasets.mesh_dataset import ObjaverseDataset, collate_fn
from functools import partial


def do_train(train_config, accelerator):
    """
    Trains a Latent Diffusion Model (LDM).
    """
    device = accelerator.device

    # --- 1. Setup Experiment Dir ---
    model_string_name = train_config['model']['model_type'].replace("/", "-")
    if train_config['train']['exp_name'] is None:
        if accelerator.is_main_process:
            os.makedirs(train_config['train']['output_dir'], exist_ok=True)
        accelerator.wait_for_everyone()
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        exp_name = f'{experiment_index:03d}-LDM-{model_string_name}'
    else:
        exp_name = train_config['train']['exp_name']
    
    checkpoint_dir = f"{train_config['train']['output_dir']}/{exp_name}/checkpoints"
    
    if accelerator.is_main_process:
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(log_dir=experiment_dir, project="MeshFlow2-LDM", name=exp_name, config=train_config)

    rank = accelerator.local_process_index

    # --- 2. Load and Freeze VAE (The First Stage) ---
    # We need the VAE to encode raw data into latents
    vae_config = train_config['vae']
    vae = AutoencoderKL(
        latent_channels=vae_config['latent_channels'], 
        decoder_type=vae_config['decoder_type'],
        num_bins=train_config['data']['num_bins'], # Needed if VAE uses quantization
        use_rmsnorm=vae_config.get('use_rmsnorm', False)
    ).to(device)

    # Load VAE weights
    if 'vae_ckpt' in vae_config and vae_config['vae_ckpt'] is not None:
        if accelerator.is_main_process:
            logger.info(f"Loading VAE from {vae_config['vae_ckpt']}")
        vae_ckpt = torch.load(vae_config['vae_ckpt'], map_location='cpu')
        # Handle state dict keys if needed
        if 'model' in vae_ckpt: vae_ckpt = vae_ckpt['model']
        vae_ckpt = {k.replace('module.', ''): v for k, v in vae_ckpt.items()}
        vae.load_state_dict(vae_ckpt, strict=False)
    else:
        raise ValueError("Must provide 'vae_ckpt' in config to train Latent Diffusion!")

    vae.eval()
    requires_grad(vae, False) # Freeze VAE
    
    # Latent Scaling Factor (Crucial for LDM)
    # Latents usually have std << 1. Diffusion expects std ~ 1.
    latent_scale_factor = vae_config.get('scale_factor', 1.0) 

    # --- 3. Create DiT Model (The Second Stage) ---
    # Note: input_dim is now latent_channels, NOT raw coordinate dim (9)
    model = DiT(
        hidden_dim=train_config['model']['hidden_dim'], 
        num_heads=train_config['model']['num_heads'],
        max_length=train_config['model']['max_length'],
        input_dim=vae_config['latent_channels'], # <--- INPUT IS LATENT
        num_layers=train_config['model']['num_layers'],
        gradient_checkpointing=train_config['model']['gradient_checkpointing'],
        use_coord_encoding=train_config['model']['use_coord_encoding'],
        version=train_config['model']['version'],
        pe_freq=train_config['model']['pe_freq'],
        mixed_precision=train_config['model']['mixed_precision'],
        use_dit_like_pe=train_config['model']['use_dit_like_pe'],
        face_cond=train_config['model']['face_cond'],
        face_bin=train_config['model']['face_bin'],
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
    )
    ema = deepcopy(model).to(device)

    # Load DiT weights if resuming or finetuning
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location='cpu')
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
    
    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[rank])
    
    # --- 4. Setup Transport (Flow Matching / Diffusion) ---
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

    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # --- 5. Data Setup ---
    # We still load raw meshes, VAE encodes them on-the-fly
    dataset = ObjaverseDataset(
        data_pth=train_config['data']['data_path'],
        training=True,
        noise_sort=train_config['data']['noise_sort'],
        use_custom_prior=train_config['data']['use_custom_prior'],
        use_decimated_dataset=train_config['data'].get('use_decimated_dataset', False),
        do_dataset_normalize=train_config['data'].get('do_dataset_normalize', False),
    )
    
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, max_seq_length=800)
    )

    # Prepare with Accelerator
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    # Resume Logic
    train_steps = 0
    if train_config['train'].get('resume', False):
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            ema.load_state_dict(checkpoint['ema'])
            opt.load_state_dict(checkpoint['opt'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming LDM from: {latest_checkpoint}")

    # Training Loop
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    while True:
        for data in loader:
            x_raw = data['tokens'] # [B, N, 9] Raw Coordinates
            y = data['num_faces']
            mask = data['masks']

            if accelerator.mixed_precision == 'no':
                x_raw = x_raw.to(device, dtype=torch.float32)
            else:
                x_raw = x_raw.to(device) # autocast handles dtype
            
            y = y.to(device)
            mask = mask.to(device)

            # --- [Step A] Encode to Latent Space ---
            with torch.no_grad():
                # 1. Encode: x -> posterior
                # Assuming VAE returns (recon, posterior, z) or just posterior
                # We need to access the encoder part. 
                # If model is AutoencoderKL, we usually do:
                # posterior = vae.encode(x_raw)
                # z = posterior.sample()
                # But your previous VAE forward might return recon, post, z. 
                # Let's assume vae.encode() exists or we run forward and take z.
                
                # Option 1: If your VAE has specific encode method:
                # posterior = vae.encode(x_raw)
                # x_latents = posterior.sample()
                
                # Option 2: Run forward (less efficient but safer if API unknown)
                _, posterior, x_latents = vae(x_raw, cond=y, mask=mask, sample_posterior=True)
                
                # 2. Scale Latents
                # Standardize distribution for diffusion
                x_latents = x_latents * latent_scale_factor

            # --- [Step B] Diffusion Training ---
            # x1 (Target) is now the Latent Vector
            # x0 (Noise) needs to match Latent shape [B, N, C]
            
            # Sample noise matching latent shape
            noise = torch.randn_like(x_latents)
            
            model_kwargs = dict(y=y, mask=mask)
            
            # Compute Flow Matching / Diffusion Loss
            loss_dict = transport.training_losses(model, x1=x_latents, x0=noise, model_kwargs=model_kwargs)
            
            loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            accelerator.backward(loss)
            
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            
            opt.step()
            update_ema(ema, model.module)

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % train_config['train']['log_every'] == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) LDM Loss: {avg_loss:.4f}, Spd: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Checkpointing
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                        "vae_config": vae_config # Save VAE config reference
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

# --- Utils ---
def load_weights_with_shape_check(model, checkpoint, rank=0):
    model_state_dict = model.state_dict()
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                if rank == 0: print(f"Shape mismatch: {name}")
    model.load_state_dict(model_state_dict, strict=False)
    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(level=logging.INFO, 
                            format='[\033[34m%(asctime)s\033[0m] %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")])
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_ldm.yaml')
    args = parser.parse_args()
    accelerator = Accelerator()
    do_train(load_config(args.config), accelerator)