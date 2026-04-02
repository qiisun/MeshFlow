import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils.logger import WandBLogger as SummaryWriter
import yaml
import numpy as np
import logging
import os
import argparse
from time import time
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from functools import partial
from accelerate import Accelerator

from models.equidit import DiT
from flow_matching import create_transport
from inference import do_sample_simple
from datasets.mesh_dataset import ObjaverseDataset, collate_fn


def setup_experiment(train_config, accelerator):
    model_string_name = train_config['model']['model_type'].replace('/', '-')
    if train_config['train']['exp_name'] is None:
        if accelerator.is_main_process:
            os.makedirs(train_config['train']['output_dir'], exist_ok=True)
        accelerator.wait_for_everyone()
        experiment_index = len(glob(f"{train_config['train']['output_dir']}/*"))
        exp_name = f'{experiment_index:03d}-{model_string_name}'
    else:
        exp_name = train_config['train']['exp_name']

    checkpoint_dir = f"{train_config['train']['output_dir']}/{exp_name}/checkpoints"
    experiment_dir = f"{train_config['train']['output_dir']}/{exp_name}"

    logger = None
    writer = None
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(log_dir=experiment_dir, project="MeshFlow2", name=exp_name, config=train_config)

    return checkpoint_dir, experiment_dir, logger, writer


def build_model_from_config(train_config):
    model_arch = train_config['model'].get('model_type', 'equidit')
    if model_arch != 'equidit':
        raise ValueError(f"Unsupported model_type: {model_arch}. Only 'equidit' is supported.")

    model = DiT(
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

    return model, model_arch


def build_dataloaders(train_config, accelerator):
    dataset = ObjaverseDataset(
        data_pth=train_config['data']['data_path'],
        training=True,
        noise_sort=train_config['data']['noise_sort'],
        use_custom_prior=False,
        do_dataset_normalize=True,
        use_rot_aug=train_config['data']['use_rot_aug'],
        use_scale_aug=train_config['data']['use_scale_aug'],
        use_permut_aug=train_config['data']['use_permute_aug'],
    )

    batch_size_per_gpu = int(np.round(train_config['train']['global_batch_size'] / accelerator.num_processes))
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn, max_seq_length=800),
    )

    valid_loader = None
    if 'valid_path' in train_config['data']:
        valid_dataset = ObjaverseDataset(
            data_pth=train_config['data']['data_path'],
            noise_sort=train_config['data']['noise_sort'],
            training=False,
            use_custom_prior=False,
            do_dataset_normalize=True,
            use_rot_aug=False,
            use_scale_aug=False,
            use_permut_aug=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, max_seq_length=800),
        )

        valid_size = len(valid_dataset)
    else:
        valid_size = 0

    return loader, valid_loader, len(dataset), valid_size, batch_size_per_gpu


def do_train(train_config, accelerator):
    """Train MeshFlow model."""
    device = accelerator.device

    checkpoint_dir, experiment_dir, logger, writer = setup_experiment(train_config, accelerator)

    rank = accelerator.local_process_index

    model, model_arch = build_model_from_config(train_config)
    
    if accelerator.is_main_process:
        logger.info(f"Using model architecture: {model_arch}")
    
    ema = deepcopy(model).to(device)

    if 'weight_init' in train_config['train']:
        checkpoint = torch.load(train_config['train']['weight_init'], map_location=lambda storage, loc: storage)
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
        use_lognorm=train_config['transport'].get('use_lognorm', False),
        use_jit=train_config['transport'].get('use_jit', False),
    )
    if accelerator.is_main_process:
        logger.info(f"LightningDiT Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f'Use lognorm sampling: {train_config["transport"]["use_lognorm"]}')
    opt = torch.optim.AdamW(model.parameters(), lr=train_config['optimizer']['lr'], weight_decay=0, betas=(0.9, train_config['optimizer']['beta2']))

    loader, valid_loader, train_size, valid_size, batch_size_per_gpu = build_dataloaders(train_config, accelerator)
    global_batch_size = batch_size_per_gpu * accelerator.num_processes

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {train_size:,} images {train_config['data']['data_path']}")
        logger.info(f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size")
        if valid_loader is not None:
            logger.info(f"Validation Dataset contains {valid_size:,} images {train_config['data']['valid_path']}")
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    
    train_config['train']['resume'] = train_config['train'].get('resume', False)
    train_steps = 0

    if train_config['train']['resume']:
        checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'])
            ema.load_state_dict(checkpoint['ema'])
            train_steps = int(latest_checkpoint.split('/')[-1].split('.')[0])
            if accelerator.is_main_process:
                logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting training from scratch.")
    model, opt, loader = accelerator.prepare(model, opt, loader)

    log_steps = 0
    running_loss = 0
    start_time = time()

    while True:
        for data in loader:
            x1 = data['tokens']
            x0 = data['noise']
            y = data['num_faces']
            mask = data['masks']

            if accelerator.mixed_precision == 'no':
                x1 = x1.to(device, dtype=torch.float32)
                x0 = x0.to(device, dtype=torch.float32)
            else:
                x1 = x1.to(device)
                x0 = x0.to(device)
            y = y.to(device)
            mask = mask.to(device)
            model_kwargs = dict(y=y, mask=mask)
                
            loss_dict = transport.training_losses(model, x1, x0, model_kwargs)
            loss = loss_dict["loss"].mean()
                
            opt.zero_grad()
            accelerator.backward(loss)
            if 'max_grad_norm' in train_config['optimizer']:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])
            opt.step()
            update_ema(ema, accelerator.unwrap_model(model), decay=0.9999)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config['train']['log_every'] == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.gather(avg_loss).mean().item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.5f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    writer.add_scalar('Loss/train', avg_loss, train_steps)
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % train_config['train']['ckpt_every'] == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:08d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                accelerator.wait_for_everyone()

                if 'valid_path' in train_config['data']:
                    if accelerator.is_main_process:
                        logger.info(f"Start evaluating at step {train_steps}")
                        val_loss, chamfer_loss, report_chamfer = do_sample_simple(ema,
                                                    valid_loader,
                                                    device,
                                                    transport,
                                                    train_config,
                                                    accelerator,
                                                    train_steps,
                                                    save_dir=experiment_dir)
                    if accelerator.is_main_process:
                        if report_chamfer:
                            logger.info(f"Validation Loss: {val_loss:.4f}, Chamfer Loss: {chamfer_loss:.6f}")
                        else:
                            logger.info(f"Validation Loss: {val_loss:.4f}")
                        writer.add_scalar('Loss/validation', val_loss, train_steps)
                        if report_chamfer:
                            writer.add_scalar('Loss/Chamfer', chamfer_loss, train_steps)
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
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
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
        config = yaml.safe_load(file)
    return config

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_jit.yaml')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    do_train(train_config, accelerator)