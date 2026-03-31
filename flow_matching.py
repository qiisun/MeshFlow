"""
Simplified Flow Matching Module (Camera Ready Version)
Linear interpolation path + Velocity prediction + L2 loss + ODE sampling
"""

import torch as th
import numpy as np
from torchdiffeq import odeint
from functools import partial

# ============================================================================
# 1. Interpolation Path: Linear Coupling Plan (x_t = t*x1 + (1-t)*x0)
# ============================================================================

def expand_t_like_x(t, x):
    """Reshape time t to broadcastable dimension of x"""
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


class LinearPath:
    """Linear interpolation path for flow matching"""
    
    @staticmethod
    def compute_xt(t, x0, x1):
        """Compute x_t = t*x1 + (1-t)*x0"""
        t = expand_t_like_x(t, x1)
        return t * x1 + (1 - t) * x0
    
    @staticmethod
    def compute_ut(t, x0, x1):
        """Compute velocity u_t = dx_t/dt = x1 - x0"""
        # For linear path, velocity is constant
        return x1 - x0
    
    @staticmethod
    def plan(t, x0, x1, timestep_shift = 0.):
        """Compute (t, x_t, u_t) for training"""
        t = expand_t_like_x(t, x0)
        t = LinearPath.compute_time_shift(t, timestep_shift=timestep_shift)
        xt = LinearPath.compute_xt(t, x0, x1)
        ut = LinearPath.compute_ut(t, x0, x1)
        return t, xt, ut

    @staticmethod
    def compute_time_shift(t_n, timestep_shift = 0.):
        if timestep_shift > 0:
            # Apply timestep shift transformation
            numerator = timestep_shift* t_n
            denominator = 1 + (timestep_shift - 1) * t_n
            return numerator / denominator
        return t_n
    

# ============================================================================
# 2. Transport: Loss computation + Drift function
# ============================================================================

def mean_flat(tensor, mask=None):
    """Average over all axes except batch"""
    if mask is not None:
        tensor = tensor * mask
        return tensor.sum() / mask.sum().clamp(min=1)
    return tensor.mean()


class FlowMatching:
    """Flow Matching trainer with velocity prediction"""
    
    def __init__(self, train_eps=0, sample_eps=0):
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.path = LinearPath()
    
    def training_losses(self, model, x1, x0, model_kwargs=None):
        """
        Compute L2 loss for velocity prediction
        Args:
            model: DiT model predicting velocity
            x1: [bs, N, 9] data tokens
            x0: [bs, N, 9] noise tokens
            model_kwargs: dict with 'mask', 'y', etc.
        Returns:
            dict with 'loss' key
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        mask = model_kwargs.get('mask', None)
        bs = x1.shape[0]
        
        # Sample random time points uniformly from [0, 1]
        t = th.rand((bs,), device=x1.device)
        
        # Compute x_t and target velocity u_t
        t_plan, xt, ut = self.path.plan(t, x0, x1)
        
        # Model predicts velocity
        v_pred = model(xt, t_plan, **model_kwargs)
        
        # L2 loss on velocity prediction
        loss = mean_flat(th.square(v_pred - ut), mask)

        return {'loss': loss}
    
    def sample(self, z, model, t0=0, t1=1, num_steps=50, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        device = z.device
        t_eval = th.linspace(t0, t1, num_steps).to(z)
    
        def ode_func(t, x):
            """ODE integrand: dx/dt = v(x, t)"""
            t_scalar = th.full((x.size(0),), float(t), device=device, dtype=x.dtype)
            return model(x, t_scalar, **model_kwargs)
        
        # Integrate ODE
        trajectory = odeint(
            ode_func, 
            z, 
            t_eval,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol
        )

        return trajectory    


# ============================================================================
# 4. Convenience Factory Functions
# ============================================================================

def create_flow_matching(*args, train_eps=0, sample_eps=0, **kwargs):
    """Create a FlowMatching object with default settings.

    Legacy transport configs may pass extra kwargs (e.g. path_type,
    prediction, loss_weight, use_cosine_loss). They are intentionally
    ignored here to keep drop-in compatibility.
    """
    # Legacy call pattern from old transport:
    # create_transport(path_type, prediction, loss_weight, train_eps, sample_eps, ...)
    if len(args) >= 5:
        train_eps = args[3]
        sample_eps = args[4]
    elif len(args) == 4:
        train_eps = args[3]

    _ = kwargs
    return FlowMatching(train_eps=train_eps, sample_eps=sample_eps)

