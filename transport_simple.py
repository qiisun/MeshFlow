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
    def plan(t, x0, x1):
        """Compute (t, x_t, u_t) for training"""
        xt = LinearPath.compute_xt(t, x0, x1)
        ut = LinearPath.compute_ut(t, x0, x1)
        return t, xt, ut


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
        if mask is not None:
            loss = mean_flat(((v_pred - ut) ** 2) * mask.unsqueeze(-1))
        else:
            loss = mean_flat((v_pred - ut) ** 2)
        
        return {'loss': loss}
    
    def get_drift_fn(self):
        """
        Get the drift function for ODE sampling
        For velocity matching: dx/dt = v_theta(x_t, t)
        """
        def drift_fn(x, t, model, **model_kwargs):
            """ODE drift: simply the model output"""
            v = model(x, t, **model_kwargs)
            return v
        
        return drift_fn


# ============================================================================
# 3. ODE Sampling
# ============================================================================

class ODESampler:
    """ODE-based sampler for inference"""
    
    def __init__(self, flow_matching, t0=0, t1=1, num_steps=50, 
                 atol=1e-6, rtol=1e-3, method='euler'):
        """
        Args:
            flow_matching: FlowMatching object
            t0, t1: time range [t0, t1]
            num_steps: number of discretization steps
            atol, rtol: tolerances for ODE solver
            method: 'euler' or 'dopri' (from torchdiffeq)
        """
        self.flow_matching = flow_matching
        self.drift_fn = flow_matching.get_drift_fn()
        self.t0 = t0
        self.t1 = t1
        self.num_steps = num_steps
        self.atol = atol
        self.rtol = rtol
        self.method = method
        
        # Create uniform time steps
        self.t = th.linspace(t0, t1, num_steps)
    
    def sample(self, z, model, cfg_scale=1.0, **model_kwargs):
        """
        Sample from ODE
        Args:
            z: [bs, N, 9] initial noise
            model: DiT model with conditional sampling support
            cfg_scale: classifier-free guidance scale
            **model_kwargs: additional args like mask, y
        Returns:
            List of trajectory: [samples_t0, samples_t1, ..., samples_tT]
        """
        device = z.device
        
        def ode_func(t, x):
            """ODE integrand: dx/dt = v(x, t)"""
            t_scalar = th.full((x.size(0),), float(t), device=device, dtype=x.dtype)
            return self.drift_fn(x, t_scalar, model, **model_kwargs)
        
        # Integrate ODE
        t_eval = self.t.to(device)
        trajectory = odeint(
            ode_func, 
            z, 
            t_eval,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol
        )
        
        return trajectory
    
    def sample_ode(self, sampling_method='euler', num_steps=None, atol=None, 
                   rtol=None, reverse=False, timestep_shift=0.0):
        """
        Convenience method to create a sampling function
        Returns a callable that takes (z, model_fn, **kwargs) and returns trajectory
        """
        if num_steps is not None:
            self.num_steps = num_steps
        if atol is not None:
            self.atol = atol
        if rtol is not None:
            self.rtol = rtol
        
        self.method = sampling_method
        self.t = th.linspace(self.t0, self.t1, self.num_steps)
        
        if timestep_shift > 0:
            # Apply timestep shift transformation
            def compute_tm(t_n, shift):
                numerator = shift * t_n
                denominator = 1 + (shift - 1) * t_n
                return numerator / denominator
            self.t = th.tensor([compute_tm(t_n, timestep_shift) for t_n in self.t])
        
        return partial(self.sample)


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


def create_ode_sampler(flow_matching, t0=0, t1=1, num_steps=50, 
                       atol=1e-6, rtol=1e-3, method='euler'):
    """Create an ODE sampler"""
    return ODESampler(flow_matching, t0=t0, t1=t1, num_steps=num_steps,
                      atol=atol, rtol=rtol, method=method)


# ============================================================================
# 5. Backward Compatibility: Adapter for existing code using transport module
# ============================================================================

class Sampler:
    """Backward-compatible sampler wrapper"""
    
    def __init__(self, flow_matching):
        self.flow_matching = flow_matching
    
    def sample_ode(self, sampling_method='euler', num_steps=50, atol=1e-6, 
                   rtol=1e-3, reverse=False, timestep_shift=0.0):
        """Create and return a sampling function"""
        sampler = create_ode_sampler(
            self.flow_matching,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            method=sampling_method
        )
        return sampler.sample_ode(
            sampling_method=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift
        )


# Backward compatibility: export common names
Transport = FlowMatching
create_transport = create_flow_matching
