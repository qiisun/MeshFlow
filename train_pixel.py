"""
Training Codes of LightningDiT together with VA-VAE.
It envolves advanced training methods, sampling methods, 
architecture design methods, computation methods. We achieve
state-of-the-art FID 1.35 on ImageNet 256x256.

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
from functools import partial
import torch.nn.functional as F
from accelerate import Accelerator

from models.equidit import DiT
from transport import create_transport
from inference_dit import do_sample_simple
from datasets.mesh_dataset import ObjaverseDataset, collate_fn


def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    # Compute exp_name for all processes (needed for checkpoint_dir)
    model_string_name = train_config['model']['model_type'].replace("/", "-")
    if train_config['train']['exp_name'] is None:
        # Only main process computes experiment_index to avoid race conditions
        # Then we'll broadcast it, but for simplicity, all processes can compute the same way
        # (race condition only matters if multiple processes create dirs simultaneously)
        if accelerator.is_main_process:
            os.makedirs(train_config['train']['output_dir'], exist_ok=True)
        accelerator.wait_for_everyone()  # Ensure directory exists before glob
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        exp_name = f'{experiment_index:03d}-{model_string_name}'
    else:
        exp_name = train_config['train']['exp_name']
    
    # Now exp_name is available to all processes
    checkpoint_dir = f"{train_config['train']['output_dir']}/{exp_name}/checkpoints"
    
    if accelerator.is_main_process:
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # Initialize wandb logger
        writer = SummaryWriter(log_dir=experiment_dir, project="MeshFlow2", name=exp_name, config=train_config)

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    model = DiT(hidden_dim=train_config['model']['hidden_dim'], # 768
                num_heads=train_config['model']['num_heads'],
                max_length=train_config['model']['max_length'],
                num_layers=train_config['model']['num_layers'],
                gradient_checkpointing = train_config['model']['gradient_checkpointing'],
                use_coord_encoding=train_config['model']['use_coord_encoding'],
                version = train_config['model']['version'],
                pe_freq=train_config['model']['pe_freq'],
                mixed_precision=train_config['model']['mixed_precision'],
                use_dit_like_pe=train_config['model']['use_dit_like_pe'],
                face_cond=train_config['model']['face_cond'],
                face_bin=train_config['model']['face_bin'],
                use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
            )
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # load pretrained model
    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
        # remove the prefix 'module.' from the keys
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        model = load_weights_with_shape_check(model, checkpoint, rank=rank)
        ema = load_weights_with_shape_check(ema, checkpoint, rank=rank)
        if accelerator.is_main_process:
            logger.info(f"Loaded pretrained model from {train_config['train']['weight_init']}")
    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
        use_jit=train_config['transport']['use_jit'] if 'use_jit' in train_config['transport'] else False,
    )  # default: velocity; 
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
        logger.info(f'Use cosine loss: {train_config["transport"]["use_cosine_loss"]}')
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    # Setup data
    dataset = ObjaverseDataset(
        data_pth=train_config['data']['data_path'],
        training=True,
        noise_sort=train_config['data']['noise_sort'],
        use_custom_prior=train_config['data']['use_custom_prior'] if 'use_custom_prior' in train_config['data'] else False,
        use_decimated_dataset=train_config['data']['use_decimated_dataset'] if 'use_decimated_dataset' in train_config['data'] else False,
        do_dataset_normalize=train_config['data']['do_dataset_normalize'] if 'do_dataset_normalize' in train_config['data'] else False,
        vae=False,
        use_rot_aug=train_config['data']['use_rot_aug'] if 'use_rot_aug' in train_config['data'] else True,
        use_scale_aug=train_config['data']['use_scale_aug'] if 'use_scale_aug' in train_config['data'] else True,
        use_permut_aug=train_config['data']['use_permut_aug'] if 'use_permut_aug' in train_config['data'] else True,
    )
    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, max_seq_length=800)
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
    
    if 'valid_path' in train_config['data']:
        valid_dataset = ObjaverseDataset(
        data_pth=train_config['data']['data_path'],
        noise_sort=train_config['data']['noise_sort'],
        training=False,
        use_custom_prior=train_config['data']['use_custom_prior'] if 'use_custom_prior' in train_config['data'] else False,
        use_decimated_dataset=train_config['data']['use_decimated_dataset'] if 'use_decimated_dataset' in train_config['data'] else False,
        do_dataset_normalize=train_config['data']['do_dataset_normalize'] if 'do_dataset_normalize' in train_config['data'] else False,
        vae=False,
        use_rot_aug=False,
        use_scale_aug=False,
        use_permut_aug=False,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=False, # should drop last false for validation
            collate_fn=partial(collate_fn, max_seq_length=800)
        )
        if accelerator.is_main_process:
            logger.info(f"Validation Dataset contains {len(valid_dataset):,} images {train_config['data']['valid_path']}")
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    train_config['train']['resume'] = train_config['train']['resume'] if 'resume' in train_config['train'] else False

    if train_config['train']['resume']:
        # check if the checkpoint exists
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getsize(x))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            # opt.load_state_dict(checkpoint['opt'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    if not train_config['train']['resume']:
        train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_checkpoint = train_config['train']['use_checkpoint'] if 'use_checkpoint' in train_config['train'] else True
    if accelerator.is_main_process:
        logger.info(f"Using checkpointing: {use_checkpoint}")

    while True:
        for data in loader:
            x1 = data['tokens']
            x0 = data['noise'] # presampled noise
            y = data['num_faces']
            mask = data['masks']

            if accelerator.mixed_precision == 'no':
                x1 = x1.to(device, dtype=torch.float32)
                x0 = x0.to(device, dtype=torch.float32)
                y = y
            else:
                x1 = x1.to(device)
                x0 = x0.to(device)
                y = y.to(device)
            model_kwargs = dict(y=y, mask=mask)
                
            loss_dict = transport.training_losses(model, x1, x0, model_kwargs)
            
            if 'cos_loss' in loss_dict:
                mse_loss = loss_dict["loss"].mean()
                loss = loss_dict["cos_loss"].mean() + mse_loss
            else:
                loss = loss_dict["loss"].mean()
                
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            if 'cos_loss' in loss_dict:
                running_loss += mse_loss.item()
            else:
                running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.5f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

                # TODO: Evaluate on validation set
                if 'valid_path' in train_config['data']:
                    if accelerator.is_main_process: # only validate on main process
                        logger.info(f"Start evaluating at step {train_steps}")
                        val_loss = do_sample_simple(model, 
                                                    valid_loader, 
                                                    device, 
                                                    transport, 
                                                    train_config, 
                                                    accelerator, 
                                                    train_steps, 
                                                    save_dir=experiment_dir)
                    # dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
                    # val_loss = val_loss.item() / dist.get_world_size()
                    if accelerator.is_main_process:
                        logger.info(f"Validation Loss: {val_loss:.4f}")
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                    model.train()
            if train_steps >= train_config['train']['max_steps']:
                break
        if train_steps >= train_config['train']['max_steps']:
            break

    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator

def load_weights_with_shape_check(model, checkpoint, rank=0):
    
    model_state_dict = model.state_dict()
    # check shape and load weights
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for x_embedder.proj.weight
                # the pretrained model is trained with 256x256 images
                # we can load the weights by resizing the weights
                # and keep the first 3 channels the same
                weight = torch.zeros_like(model_state_dict[name])
                weight[:, :16] = param[:, :16]
                model_state_dict[name] = weight
            else:
                if rank == 0:
                    print(f"Skipping loading parameter '{name}' due to shape mismatch: "
                        f"checkpoint shape {param.shape}, model shape {model_state_dict[name].shape}")
        else:
            if rank == 0:
                print(f"Parameter '{name}' not found in model, skipping.")
    # load state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    return model

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)