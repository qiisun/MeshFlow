import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from tqdm import tqdm
from functools import partial
import warnings

warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")

from models.equivae import loss_vae, AutoencoderKL, index_to_float
from datasets.mesh_dataset import ObjaverseDataset, collate_fn
from datasets.mesh_dataset import save_mesh


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    base_lr = optimizer.param_groups[0]['lr']
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max(0.0, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)

def do_train(train_config, accelerator):
    """
    Trains a LightningDiT.
    """
    # Setup accelerator:
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config['train']['output_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        model_string_name = train_config['model']['model_type'].replace("/", "-")
        if train_config['train']['exp_name'] is None:
            exp_name = f'{experiment_index:03d}-{model_string_name}'
        else:
            exp_name = train_config['train']['exp_name']
        experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        tensorboard_dir_log = f"tensorboard_logs/{exp_name}"
        os.makedirs(tensorboard_dir_log, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir_log)

        # add configs to tensorboard
        config_str=json.dumps(train_config, indent=4)
        writer.add_text('training configs', config_str, global_step=0)
    checkpoint_dir = f"{train_config['train']['output_dir']}/{train_config['train']['exp_name']}/checkpoints"

    # get rank
    rank = accelerator.local_process_index

    # Create model:
    model = AutoencoderKL(latent_channels=4)
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
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))
    
    scheduler = get_cosine_schedule_with_warmup(opt, 500, train_config['train']['max_steps'])
    
    # Setup data
    dataset = ObjaverseDataset(
        data_pth=train_config['data']['data_path'],
        training=True,
        noise_sort=train_config['data']['noise_sort'],
        use_custom_prior=train_config['data']['use_custom_prior'] if 'use_custom_prior' in train_config['data'] else False,
        use_decimated_dataset=train_config['data']['use_decimated_dataset'] if 'use_decimated_dataset' in train_config['data'] else False,
        do_dataset_normalize=train_config['data']['do_dataset_normalize'] if 'do_dataset_normalize' in train_config['data'] else False,
        vae=True,
        overfit=train_config['data']['overfit'] if 'overfit' in train_config['data'] else False,
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
        vae=True,
        overfit=train_config['data']['overfit'] if 'overfit' in train_config['data'] else False,
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
            y = data['num_faces']
            mask = data['masks']

            if accelerator.mixed_precision == 'no':
                x1 = x1.to(device, dtype=torch.float32)
                y = y
            else:
                x1 = x1.to(device)
                y = y.to(device)
            recon, posterior, z = model(x1, cond=y, mask=mask)        
            loss, rec_l, kl_l = loss_vae(x1, recon, posterior, mask=mask, kl_weight=train_config['train']['kl_weight'])
            # loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            scheduler.step()
            update_ema(ema, model.module)

            # Log loss values:
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
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
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
                        
                        total_val_loss = 0.0
                        total_rec_loss = 0.0
                        num_batches = 0
                        
                        save_mesh_dir = f'{experiment_dir}/mesh_{train_steps}'
                        os.makedirs(save_mesh_dir, exist_ok=True)
                        for i, data in tqdm(enumerate(valid_loader), desc="Generating Demo Samples"):
                            x1 = data['tokens'].to(device)
                            y = data['num_faces'].to(device)
                            mask = data['masks'].to(device)
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                with torch.no_grad():
                                    recon, posterior, z  = ema(x1, cond=y, mask=mask, sample_posterior=False)
                                    val_loss, rec_loss, _ = loss_vae(x1, recon, posterior, mask=mask, kl_weight=train_config['train']['kl_weight'], num_bins=train_config['data']['num_bins'])
                                    total_val_loss += val_loss.item()
                                    total_rec_loss += rec_loss.item()
                                    num_batches += 1
                                    
                                    if i < 5:
                                        if train_config['model']['decoder_type'] == 'cls':
                                            recon_indices = torch.argmax(recon, dim=-1)
                                            recon_coords = index_to_float(
                                                recon_indices, 
                                                min_val=-0.5, 
                                                max_val=0.5, 
                                                num_bins=train_config['data']['num_bins']
                                            )
                                            save_mesh(recon_coords[0].to(torch.float32).cpu().numpy(), f'{save_mesh_dir}/{i:03d}.obj')

                                        if train_config['model']['decoder_type'] == 'reg':
                                            save_mesh(recon[0].to(torch.float32).cpu().numpy(), f'{save_mesh_dir}/{i:03d}.obj')

                        if num_batches > 0:
                            avg_val_loss = total_val_loss / num_batches
                            avg_rec_loss = total_rec_loss / num_batches
                        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                        writer.add_scalar('Loss/validation', avg_val_loss, train_steps)
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