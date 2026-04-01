"""
Simplified Flow Matching Module (Camera Ready Version)
Linear interpolation path + Velocity prediction + L2 loss + ODE sampling
"""

import torch as th
import numpy as np
from torchdiffeq import odeint

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
        mask = mask.to(dtype=tensor.dtype)
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(-1)
        masked = tensor * mask
        denom = mask.expand_as(tensor).sum().clamp(min=1)
        return masked.sum() / denom
    return tensor.mean()


class FlowMatching:
    """Flow Matching trainer with velocity prediction"""
    
    def __init__(
        self,
        train_eps=0,
        sample_eps=0,
        use_lognorm=False,
        lognorm_mean=0.0,
        lognorm_std=1.0,
        use_jit=False,
        prediction='velocity',
    ):
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_lognorm = use_lognorm
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std
        self.use_jit = use_jit
        self.prediction = prediction
        self.path = LinearPath()

    def _eps(self, value, default=1e-5):
        if value is None:
            return default
        if isinstance(value, (float, int)) and value > 0:
            return float(value)
        return default

    def _is_x1_prediction(self):
        if self.use_jit:
            return True
        pred = str(self.prediction).lower()
        return pred in {'x1', 'data', 'endpoint', 'x_start'}

    def _to_velocity(self, model_out, x_t, t, eps):
        if not self._is_x1_prediction():
            return model_out
        denom = (1 - t).clamp_min(eps)
        return (model_out - x_t) / denom

    def sample_timesteps(self, bs, device):
        if self.use_lognorm:
            # SD3-style logit-normal-like sampling: t = sigmoid(N(mean, std)).
            z = th.randn((bs,), device=device) * self.lognorm_std + self.lognorm_mean
            return th.sigmoid(z)
        return th.rand((bs,), device=device)
    
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
        
        # Sample random time points from configured schedule in [0, 1].
        t = self.sample_timesteps(bs, x1.device)
        
        # Compute x_t and target velocity u_t
        t_plan, xt, ut = self.path.plan(t, x0, x1)
        
        # Model predicts velocity
        model_out = model(xt, t_plan, **model_kwargs)
        v_pred = self._to_velocity(model_out, xt, t_plan, eps=self._eps(self.train_eps))
        
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
            model_out = model(x, t_scalar, **model_kwargs)
            t_broadcast = expand_t_like_x(t_scalar, x)
            return self._to_velocity(model_out, x, t_broadcast, eps=self._eps(self.sample_eps))
        
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

def create_transport(*args, train_eps=0, sample_eps=0, **kwargs):
    """Create a FlowMatching object with default settings.

    Legacy transport configs may pass extra kwargs (e.g. path_type,
    prediction, loss_weight). They are intentionally
    ignored here to keep drop-in compatibility.
    """
    # Legacy call pattern from old transport:
    # create_transport(path_type, prediction, loss_weight, train_eps, sample_eps, ...)
    if len(args) >= 5:
        train_eps = args[3]
        sample_eps = args[4]
    elif len(args) == 4:
        train_eps = args[3]

    prediction = args[1] if len(args) >= 2 else kwargs.get('prediction', 'velocity')

    use_lognorm = kwargs.get('use_lognorm', False)
    lognorm_mean = kwargs.get('lognorm_mean', 0.0)
    lognorm_std = kwargs.get('lognorm_std', 1.0)
    use_jit = kwargs.get('use_jit', False)

    return FlowMatching(
        train_eps=train_eps,
        sample_eps=sample_eps,
        use_lognorm=use_lognorm,
        lognorm_mean=lognorm_mean,
        lognorm_std=lognorm_std,
        use_jit=use_jit,
        prediction=prediction,
    )