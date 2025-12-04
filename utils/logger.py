"""
Weights & Biases (wandb) logger with TensorBoard SummaryWriter-like API.
This module provides a drop-in replacement for TensorBoard's SummaryWriter.
"""

import wandb
import os
from typing import Optional, Union
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

