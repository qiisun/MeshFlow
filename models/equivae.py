import re
import sys
sys.path.append('.')
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from models.utils import get_2d_sincos_pos_embed
from models.equidit import LabelEmbedder, XEmbedder, VertexCrossAttention, DiTLayer, FinalLayer


def float_to_index(value, min_val=-0.5, max_val=0.5, num_bins=512):
    norm = (value - min_val) / (max_val - min_val)
    norm = torch.clamp(norm, 0, 1)
    return (norm * (num_bins - 1)).long()


def float_to_index_np(value, min_val=-0.5, max_val=0.5, num_bins=512):
    norm = (value - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    return (norm * (num_bins - 1)).astype(int)

def index_to_float(index, min_val=-0.5, max_val=0.5, num_bins=512):
    norm = index.float() / (num_bins - 1)
    value = norm * (max_val - min_val) + min_val
    return value

def index_to_float_np(index, min_val=-0.5, max_val=0.5, num_bins=512):
    norm = index.astype(np.float32) / (num_bins - 1)
    value = norm * (max_val - min_val) + min_val
    return value

class AutoencoderKL(nn.Module):
    def __init__(self, 
                 hidden_dim=768,
                 num_layers=6,
                num_heads=12,
                 latent_channels=3,
                 decoder_type="reg",
                 num_bins=256,
                 use_rmsnorm=False,
                 face_bin=10,
                 deterministic=False,
                 use_identity_encoder=False,
                 fixed_std=0.0):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_bins = num_bins
        
        self.use_identity_encoder = use_identity_encoder
        self.fixed_std = fixed_std
        self.deterministic = deterministic

           
        if self.use_identity_encoder:
            print(f"[AutoencoderKL] Using Identity/Linear Encoder with Fixed Std: {self.fixed_std}")
            self.encoder = None 
            self.quant_linear = None 
            self.input_proj = nn.Identity()
        else:
            self.encoder = Model(hidden_dim=hidden_dim, model_type='encoder', num_bins=num_bins, use_rmsnorm=use_rmsnorm, face_bin=face_bin,
                                 num_layers=num_layers,
                             num_heads=num_heads) # encoder 
            self.quant_linear = nn.Linear(hidden_dim, 2 * latent_channels) # 9 -> 4*2
        
        self.post_quant_linear = nn.Linear(latent_channels, hidden_dim)
        self.decoder = Model(hidden_dim=hidden_dim, model_type='decoder', decoder_type=decoder_type, num_bins=num_bins, use_rmsnorm=use_rmsnorm, face_bin=face_bin,
                             use_identity_encoder=use_identity_encoder,
                             num_layers=num_layers,
                             num_heads=num_heads,) # decoder   
        
    def encode(self, x, cond, mask):
        """
        Input: [B, N, 9]
        Output Posterior over: [B, 3*N, latent_dim]
        """
        if self.use_identity_encoder:
            b, n, _ = x.shape
            x = x.view(b, -1, 3) # [B, N*3, 3]  
            mean = self.input_proj(x) # [B, N, latent_channels]
            if self.fixed_std > 0:
                fixed_logvar_val = 2 * math.log(self.fixed_std)
                logvar = torch.full_like(mean, fixed_logvar_val)
            else:
                logvar = torch.full_like(mean, -20.0) 
            moments = torch.cat([mean, logvar], dim=-1)
        else:
            h = self.encoder(x, cond, mask)
            moments = self.quant_linear(h)
            
        mask_expanded = mask.repeat_interleave(3, dim=1)
        posterior = DiagonalGaussianDistribution(moments, mask=mask_expanded, deterministic=self.deterministic)
        return posterior
    
    def decode(self, z, cond, mask):
        """
        解码过程：输入 Latent -> 重建图像
        """
        z = self.post_quant_linear(z)
        dec = self.decoder(z, cond, mask)
        return dec

    def forward(self, input, cond, mask=None, sample_posterior=True):
        """
        端到端的前向传播：包含 Loss 计算
        这是训练时调用的主函数
        """
        # 1. Encode
        posterior = self.encode(input, cond, mask)
        
        # 2. Sample (训练时采样，推理时通常取 mode)
        if sample_posterior:
            z = posterior.sample() # [b, N, c] 
        else:
            z = posterior.mode() # z[mask.unsqueeze(2).repeat(1,1,3).reshape(bs, -1)]
        reconstruction = self.decode(z, cond, mask)
        
        if self.use_identity_encoder:
            z = z.reshape(reconstruction.shape[0], reconstruction.shape[1], -1) # [B, N*3, 3]
            reconstruction += z
        return reconstruction, posterior, z

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, mask=None, deterministic=False):
        """
        parameters: [B, N, 2*C]
        mask: [B, N] or [B, N, 1] 
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -20.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.mask = mask

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        if self.mask is not None:
            if self.mask.dim() == 2:
                m = self.mask.unsqueeze(-1)
            else:
                m = self.mask
            x = x * m 
        return x

    def mode(self):
        x = self.mean
        if self.mask is not None:
            if self.mask.dim() == 2:
                m = self.mask.unsqueeze(-1)
            else:
                m = self.mask
            x = x * m
        return x

    def kl(self):
        kl_val = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
        if self.mask is not None:
            if self.mask.dim() == 2:
                m = self.mask.unsqueeze(-1)
            else:
                m = self.mask
            kl_val = kl_val * m
        return kl_val

# modelify from equiDIT
class Model(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, max_length=800,
                 num_layers=6, gradient_checkpointing=True, 
                 class_dropout_prob=0.1, use_coord_encoding=True, 
                 version=3, pe_freq=20, mixed_precision='bf16',
                 use_dit_like_pe=False, face_cond=True, face_bin=100,
                 use_rmsnorm=False, 
                 model_type='encoder',
                 decoder_type="reg",
                 num_bins=512,
                 use_identity_encoder=False):
        super().__init__()
        self.model_type = model_type
        self.decoder_type = decoder_type
        self.use_dit_like_pe = use_dit_like_pe
        self.kl_weight = 1e-6
        self.use_l1_loss = True
        self.use_identity_encoder = use_identity_encoder

        input_dim = 3 if version > 1 else 9
        self.version = version
        self.use_coord_encoding = use_coord_encoding
        self.hidden_size = hidden_dim
        # project input
        if model_type == 'encoder':
            self.x_embedder = XEmbedder(hidden_dim, pe_freq=pe_freq, use_coord_encoding=use_coord_encoding, version=version)
            
        # positional encoding (just use a learnable positional encoding)
        if use_dit_like_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) / hidden_dim ** 0.5)
        
        # timestep encoding
        self.face_cond = face_cond
        if face_cond:
            self.uncond_y = max_length + 1
            self.num_classes = max_length//face_bin
            self.y_embedder = LabelEmbedder(max_length//face_bin, hidden_dim, class_dropout_prob) # 16+1
            self.face_bin = face_bin

        # transformer layers
        self.layers = nn.ModuleList([DiTLayer(hidden_dim, num_heads, gradient_checkpointing, mixed_precision=mixed_precision, version=version, use_rmsnorm=use_rmsnorm) for _ in range(num_layers)])

        # project out        
        modulation_type = "mult" 
        self.vertex_cross_attn = VertexCrossAttention(
            modulation_type=modulation_type, hidden_dim=hidden_dim)
        if model_type == 'decoder':
            if decoder_type == "cls":
                self.num_bins = num_bins
                self.final_layer = FinalLayer(hidden_dim, input_dim*num_bins, use_rmsnorm=use_rmsnorm) # input_dim: 9 for v2, 3 for v3
            elif decoder_type == "reg":
                self.final_layer = FinalLayer(hidden_dim, input_dim, use_rmsnorm=use_rmsnorm) # input_dim: 9 for v2, 3 for v3
        self.initialize_weights()
    

    def initialize_weights(self):
        # Initialize transformer layers:
        # 对于VAE来说，不能0初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.use_dit_like_pe:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 1)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table:
        if self.face_cond:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize embedding MLP:   
        if self.model_type == "encoder":     
            nn.init.normal_(self.x_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.x_embedder.mlp[2].weight, std=0.02)
        
        if self.use_identity_encoder:
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, y, mask=None):
        # x: [B, N, C], hidden states
        # c: [B, M, C]
        # y: [B,], face number
        # mask: [B, N]
        # return: [B, N, C], updated hidden states
        if self.model_type == 'decoder':
            B, _, C = x.shape
            x = x.view(B, -1, 3, C)
            N = x.shape[1]
            x_ = x.view(-1, 3, C) # keep the order 
            x = x.mean(dim=2) # pooled face feature
        else:
            B, N, _ = x.shape
            if self.version > 1:
                x, x_ = self.x_embedder(x) # [B, N, C], [B*N, 3, C]
            else:
                x = self.x_embedder(x) # [B, N, C]
            
        # positional encoding (default: None)
        if self.use_dit_like_pe: 
            x = x + self.pos_embed[:, :N, :]

        # only face encoding
        if self.face_cond:
            y = self.y_embedder(y // self.face_bin, self.training)
            c = y

        # transformer layers
        if self.version > 2: # v3
            x = x_.reshape(B, N*3, -1) # [B, N*3, C]
            
        for layer in self.layers:
            x = layer(x, c, mask=mask)
        
        # project out
        if self.version == 2:
            x = x.view(B*N, 1, self.hidden_size)
            x = self.vertex_cross_attn(x_, x) # [B*N, 3, C]
            x = x.view(B, N*3, -1)
        
        if self.model_type == 'encoder':
            return x # [b, N*3, C]
        elif self.model_type == 'decoder':
            x = self.final_layer(x, c) # v2: [B, N*3, 3]
            if self.decoder_type == "cls":
                x = x.view(B, N, 9, self.num_bins) # [B, N, 9, num_bins]
            elif self.decoder_type == "reg":
                x = x.view(B, N, 9) # [B, N, 9, num_bins]
            return x

def _masked_mean(x, mask):
    if mask is None: 
        return x.mean()
    if x.dim() > mask.dim(): 
        mask = mask.unsqueeze(-1)
    if x.shape[1] != mask.shape[1]:
         scale = x.shape[1] // mask.shape[1]
         mask = mask.repeat_interleave(scale, dim=1)
    return (x * mask).sum() / (mask.expand_as(x).sum() + 1e-8)

def compute_face_normals(vertices):
    B, N, _ = vertices.shape
    tris = vertices.view(B, N, 3, 3)
    
    v0 = tris[:, :, 0, :]
    v1 = tris[:, :, 1, :]
    v2 = tris[:, :, 2, :]
    
    e1 = v1 - v0
    e2 = v2 - v0
    
    normals = torch.cross(e1, e2, dim=-1)
    normals = F.normalize(normals, dim=-1, p=2, eps=1e-6)
    return normals

# @torch.jit.script
def weighting_from_triangle_soup(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    x1 = inputs[..., 3] - inputs[..., 0]
    y1 = inputs[..., 4] - inputs[..., 1]
    z1 = inputs[..., 5] - inputs[..., 2]
    
    x2 = inputs[..., 6] - inputs[..., 0]
    y2 = inputs[..., 7] - inputs[..., 1]
    z2 = inputs[..., 8] - inputs[..., 2]
    cx = y1 * z2 - z1 * y2
    cy = z1 * x2 - x1 * z2
    cz = x1 * y2 - y1 * x2
    area = 0.5 * torch.sqrt(cx*cx + cy*cy + cz*cz + 1e-8)
    weight = 1 / (1e-7+torch.sqrt(area))
    return weight * mask

def loss_vae(inputs, recon, posterior, mask=None, kl_weight=0, decoder_type="reg", num_bins=512,
             normal_weight=1e-3,
             loss_type="mse",
             fixed_std=0.0,
             delta=8e-3,
             weighting=None):
    if weighting is None:
        weighting = weighting_from_triangle_soup(inputs, mask)
    
    diff = inputs - recon
    rec_diff_abs = torch.abs(diff)   # L1 (MAE) base
    mse_elementwise = diff ** 2      # L2 (MSE) base
    mae_metric = _masked_mean(rec_diff_abs, mask)
    mse_metric = _masked_mean(mse_elementwise, mask)
    rmse_metric = torch.sqrt(mse_metric)
    weighted_abs_diff = rec_diff_abs * weighting[..., None]
    w_mae_metric = _masked_mean(weighted_abs_diff, mask)
    if loss_type == 'mae':
        pixel_loss = rec_diff_abs
        
    elif loss_type == "mse":
        pixel_loss = mse_elementwise / (fixed_std)
        
    elif loss_type == "weighted_mse":
        pixel_loss = mse_elementwise * weighting[..., None]
        
    elif loss_type == "weighted_mae":
        pixel_loss = rec_diff_abs * weighting[..., None]
        
    elif loss_type == "huber":
        huber_loss = torch.where(
            rec_diff_abs < delta,
            0.5 * mse_elementwise,
            delta * (rec_diff_abs - 0.5 * delta)
        )
        pixel_loss = huber_loss
        
    elif loss_type == "berhu": 
        berhu_loss = torch.where(
            rec_diff_abs <= delta,
            rec_diff_abs,
            (mse_elementwise + delta**2) / (2 * delta)
        )
        pixel_loss = berhu_loss
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")    
    rec_loss = _masked_mean(pixel_loss, mask)
    kl_vals = posterior.kl() # [B, N] or [B, N, C]
    kl_loss = _masked_mean(kl_vals, mask)
    total_loss = rec_loss + kl_weight * kl_loss
    
    return total_loss, kl_loss, mae_metric, rmse_metric, w_mae_metric

if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, c = 800, 9
    # model = AutoencoderKL(latent_channels=4).cuda()
    model = AutoencoderKL(
        latent_channels=3,         
        use_identity_encoder=True, 
        fixed_std=0.01         
    ).to(device)

    x = torch.randn(2, n, c).cuda() # Batch=2
    cond = torch.randint(0, 16, (2,)).cuda() # Batch=2
    mask = torch.ones(2, n).bool().cuda()
    weighting = torch.randn(2, n).cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        recon, posterior, z = model(x, cond=cond, mask=mask)

    loss, rec_l, kl_l, rmse, w_mae_metric = loss_vae(x, recon, posterior, mask=mask, loss_type='mse',
                                                     weighting = weighting)
    print(f"Loss: {loss}, Rec Loss: {rec_l}, KL Loss: {kl_l}, RMSE: {rmse}")
    model.eval()