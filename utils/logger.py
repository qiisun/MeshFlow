"""
Weights & Biases (wandb) logger with TensorBoard SummaryWriter-like API.
This module provides a drop-in replacement for TensorBoard's SummaryWriter.
"""

import wandb
import os
import torch
import numpy as np
from typing import Optional, Union, Dict, List
import json


class WandBLogger:
    """
    A wandb logger with a TensorBoard SummaryWriter-like API.
    
    This class provides a similar interface to torch.utils.tensorboard.SummaryWriter
    but uses Weights & Biases for logging instead.
    
    Example usage:
        writer = WandBLogger(log_dir="experiments/my_exp", project="my_project")
        writer.add_scalar('Loss/train', 0.5, 100)
        writer.add_text('training configs', config_string, 0)
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        **wandb_init_kwargs
    ):
        """
        Initialize the WandB logger.
        
        Args:
            log_dir: Directory for logs (used to derive experiment name if name not provided).
                     Similar to TensorBoard's log_dir parameter.
            project: W&B project name. If not provided, defaults to 'MeshFlow2' or uses log_dir.
            name: Run name. If not provided, uses the last part of log_dir.
            config: Configuration dictionary to log to wandb.
            **wandb_init_kwargs: Additional arguments passed to wandb.init()
        """
        # Extract experiment name from log_dir if name not provided
        if name is None and log_dir is not None:
            name = os.path.basename(os.path.normpath(log_dir))
        
        # Set default project if not provided
        if project is None:
            project = "MeshFlow2"
        
        # Initialize wandb
        # If already initialized (e.g., in a distributed setting), use the existing run
        if wandb.run is None:
            # Set mode to online by default if not specified
            # The training scripts already check is_main_process before creating the logger
            init_kwargs = {
                'project': project,
                'name': name,
                'dir': log_dir,
                'mode': wandb_init_kwargs.get('mode', 'online'),  # Default to online
                **wandb_init_kwargs
            }
            if config is not None:
                init_kwargs['config'] = config
            
            wandb.init(**init_kwargs)
        
        self.log_dir = log_dir
        self.project = project
        self.name = name
        self._step = 0
        
    def add_scalar(self, tag: str, scalar_value: Union[float, int], global_step: Optional[int] = None):
        """
        Add a scalar value to the logger.
        
        Args:
            tag: Name for the scalar (e.g., 'Loss/train')
            scalar_value: Scalar value to log
            global_step: Step number. If None, uses internal step counter.
        """
        if global_step is None:
            global_step = self._step
        
        # Convert tensor to python scalar if needed
        if hasattr(scalar_value, 'item'):
            scalar_value = scalar_value.item()
        
        # Log to wandb using the tag as the metric name
        # Only log if wandb is initialized (safety check)
        if wandb.run is not None:
            wandb.log({tag: scalar_value}, step=global_step)
        
        if global_step == self._step:
            self._step += 1
    
    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None):
        """
        Add text to the logger.
        
        Args:
            tag: Name for the text (e.g., 'training configs')
            text_string: Text content to log
            global_step: Step number. If None, uses internal step counter.
        """
        if global_step is None:
            global_step = self._step
        
        # Only log if wandb is initialized (safety check)
        if wandb.run is not None:
            # Log as a table or text artifact, or as a config entry if at step 0
            if global_step == 0:
                # For initial config, we can log it as part of the config or as a summary
                # Using a text summary approach
                try:
                    # Try to parse as JSON if possible
                    config_dict = json.loads(text_string)
                    wandb.config.update(config_dict)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, log as a text summary
                    wandb.summary[f"{tag}"] = text_string
            else:
                # For later text logs, use wandb's text logging
                wandb.log({tag: wandb.Html(f"<pre>{text_string}</pre>")}, step=global_step)
        
        if global_step == self._step:
            self._step += 1
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: Optional[int] = None):
        """
        Add multiple scalars to the logger.
        
        Args:
            main_tag: Main tag prefix (e.g., 'Loss')
            tag_scalar_dict: Dictionary of tag: value pairs
            global_step: Step number. If None, uses internal step counter.
        """
        if global_step is None:
            global_step = self._step
        
        # Build the log dictionary with full tag names
        log_dict = {}
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}" if main_tag else tag
            if hasattr(value, 'item'):
                value = value.item()
            log_dict[full_tag] = value
        
        # Only log if wandb is initialized (safety check)
        if wandb.run is not None:
            wandb.log(log_dict, step=global_step)
        
        if global_step == self._step:
            self._step += 1
    
    def add_histogram_loss_by_timestep(
        self, 
        timesteps: torch.Tensor, 
        losses: torch.Tensor, 
        global_step: Optional[int] = None,
        num_bins: int = 20,
        tag: str = 'Loss/by_timestep'
    ):
        """
        Log a histogram of loss values binned by timestep/noise schedule.
        
        Args:
            timesteps: Tensor of timesteps [batch_size] or [batch_size, ...]
            losses: Tensor of loss values per sample [batch_size] or [batch_size, ...]
            global_step: Step number. If None, uses internal step counter.
            num_bins: Number of bins for the histogram
            tag: Tag name for the histogram
        """
        if global_step is None:
            global_step = self._step
        
        if wandb.run is None:
            return
        
        # Flatten if needed and convert to numpy
        timesteps_flat = timesteps.detach().cpu().flatten().numpy()
        losses_flat = losses.detach().cpu().flatten().numpy()
        
        # Create bins for timesteps
        t_min, t_max = timesteps_flat.min(), timesteps_flat.max()
        if t_max - t_min < 1e-6:  # All timesteps are the same
            bins = np.linspace(t_min - 0.1, t_max + 0.1, num_bins + 1)
        else:
            bins = np.linspace(t_min, t_max, num_bins + 1)
        
        # Bin the timesteps and compute average loss per bin
        bin_indices = np.digitize(timesteps_flat, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_losses = []
        bin_counts = []
        
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_losses.append(losses_flat[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_losses.append(0.0)
                bin_counts.append(0)
        
        # Filter out empty bins and create data for logging
        valid_bins = []
        for center, loss, count in zip(bin_centers, bin_losses, bin_counts):
            if count > 0:
                valid_bins.append((center, loss, count))
        
        if len(valid_bins) > 0:
            # Extract data
            centers, losses_avg, counts = zip(*valid_bins)
            
            # Create a table for visualization (wandb will create a nice plot from this)
            table_data = [[float(c), float(l), int(cnt)] for c, l, cnt in valid_bins]
            table = wandb.Table(
                columns=["timestep", "avg_loss", "sample_count"],
                data=table_data
            )
            # Log table - wandb will auto-create visualizations
            wandb.log({f"{tag}": wandb.plot.line(table, "timestep", "avg_loss", 
                                                title=f"Loss vs Timestep (Step {global_step})")}, 
                     step=global_step)
            
            # Also create a histogram showing distribution of losses across timesteps
            # Create histogram data from the raw loss values, weighted by timestep bins
            all_losses_for_hist = []
            for i, (center, loss_avg, count) in enumerate(valid_bins):
                # Add loss values for this timestep bin (use average, repeated by count for weighting)
                all_losses_for_hist.extend([loss_avg] * min(int(count), 1000))
            
            if len(all_losses_for_hist) > 0:
                hist = wandb.Histogram(np.array(all_losses_for_hist))
                wandb.log({f"{tag}_distribution": hist}, step=global_step)
    
    def add_gradient_norms(
        self,
        model: torch.nn.Module,
        global_step: Optional[int] = None,
        prefix: str = "Gradients"
    ):
        """
        Log gradient norms for all parameters (excluding biases).
        
        Args:
            model: PyTorch model
            global_step: Step number. If None, uses internal step counter.
            prefix: Prefix for the log tags
        """
        if global_step is None:
            global_step = self._step
        
        if wandb.run is None:
            return
        
        total_norm = 0.0
        param_norms = {}
        
        # Compute gradient norms for each parameter (excluding biases)
        for name, param in model.named_parameters():
            if param.grad is not None and 'bias' not in name.lower():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                # Store norm per parameter
                param_norms[name] = param_norm.item()
        
        total_norm = total_norm ** (1. / 2)
        
        # Log total gradient norm
        wandb.log({f"{prefix}/total_norm": total_norm}, step=global_step)
        
        # Log per-layer gradient norms (limit to avoid too many metrics)
        # Group by layer name prefix to reduce clutter
        layer_norms = {}
        for name, norm in param_norms.items():
            # Extract layer name (e.g., "blocks.0.attn.qkv_proj" -> "blocks.0.attn")
            parts = name.split('.')
            if len(parts) > 1:
                layer_name = '.'.join(parts[:-1])
            else:
                layer_name = name
            
            if layer_name not in layer_norms:
                layer_norms[layer_name] = []
            layer_norms[layer_name].append(norm)
        
        # Log average norm per layer
        for layer_name, norms in layer_norms.items():
            avg_norm = np.mean(norms)
            # Clean up layer name for wandb (remove special characters)
            clean_name = layer_name.replace('.', '/')
            wandb.log({f"{prefix}/layer_norm/{clean_name}": avg_norm}, step=global_step)
        
        # Create a histogram of all parameter gradient norms
        if param_norms:
            norms_list = list(param_norms.values())
            hist = wandb.Histogram(norms_list)
            wandb.log({f"{prefix}/param_norm_histogram": hist}, step=global_step)
    
    def close(self):
        """Finish the wandb run."""
        if wandb.run is not None:
            wandb.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Alias for backward compatibility if someone wants to import it as SummaryWriter
SummaryWriter = WandBLogger

