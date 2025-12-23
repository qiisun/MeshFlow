import sys
sys.path.append('.')
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from models.dit import LabelEmbedder, FinalLayer, TransformerBlock


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
                 latent_channels=3,
                 decoder_type="reg",
                 num_bins=256,
                 use_rmsnorm=False,
                 use_encoder=True,
                 latent_noise_std=0.02,
                 use_constant_latent_noise=True,
                 raw_noise_std=0.02):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_bins = num_bins
        self.use_encoder = use_encoder
        self.latent_noise_std = latent_noise_std
        self.use_constant_latent_noise = use_constant_latent_noise
        self.raw_noise_std = raw_noise_std
        self.quant_linear = nn.Linear(hidden_dim, 2 * latent_channels) # 9 -> 4*2
        self.post_quant_linear = nn.Linear(latent_channels, hidden_dim)   
        self.raw_to_latent = nn.Linear(9, latent_channels)
        
        self.encoder = Model(in_channels=9, model_type='encoder') # encoder 
        self.decoder = Model(in_channels=9, model_type='decoder') # decoder   
        
    def encode(self, x, cond, mask):
        """
        Input: [B, N, 9]
        Output Posterior over: [B, 3*N, latent_dim]
        """
        if not self.use_encoder:
            return None
        h = self.encoder(x, cond, mask)
        moments = self.quant_linear(h)
        mask_expanded = mask # .repeat_interleave(1, dim=1)

        posterior = DiagonalGaussianDistribution(
            moments,
            mask=mask_expanded,
            fixed_std=self.latent_noise_std if self.use_constant_latent_noise else None
        )
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
        
        if posterior is None:
            # 直接对原始 mesh 加噪声，再映射到 latent
            if self.raw_noise_std is not None and self.raw_noise_std > 0:
                noised = input + torch.randn_like(input) * self.raw_noise_std
            else:
                noised = input
            z = self.raw_to_latent(noised)
        else:
            # 2. Sample (训练时采样，推理时通常取 mode)
            if sample_posterior:
                z = posterior.sample() # [b, N, c] 
            else:
                z = posterior.mode() # z[mask.unsqueeze(2).repeat(1,1,3).reshape(bs, -1)]
        reconstruction = self.decode(z, cond, mask)
        return reconstruction, posterior, z

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, mask=None, deterministic=False, fixed_std=None):
        """
        parameters: [B, N, 2*C]
        mask: [B, N] or [B, N, 1] 
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.fixed_std = fixed_std
        if fixed_std is not None:
            self.std = torch.full_like(self.mean, fixed_std)
            self.var = self.std ** 2
            self.logvar = torch.log(self.var + 1e-8)
        else:
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
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=1,
        dim=768,
        n_layers=6,
        n_heads=12,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=1000,
        model_type='encoder',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.model_type = model_type
        if model_type == 'encoder':
            self.init_conv_seq = nn.Linear(in_channels * patch_size * patch_size, dim // 2, bias=True) # 9
            self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
            nn.init.constant_(self.x_embedder.bias, 0)
        else: # 'decoder'
            self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )

        self.freqs_cis = Model.precompute_freqs_cis(dim // n_heads, 4096)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def unpatchify(self, x):
        # c = self.out_channels
        # p = self.patch_size
        # h = w = int(x.shape[1] ** 0.5)
        # x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # x = torch.einsum("nhwpqc->nchpwq", x)
        # imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        # return imgs
        return x

    def patchify(self, x):
        # B, L, C = x.size()
        # x = x.view(
        #     B,
        #     C,
        #     H // self.patch_size,
        #     self.patch_size,
        #     W // self.patch_size,
        #     self.patch_size,
        # )
        # x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2) # [b, L, c]
        return x

    def forward(self, x, y, mask=None):
        self.freqs_cis = self.freqs_cis.to(x.device)
        if self.model_type == 'encoder':
            x = self.init_conv_seq(x)
            x = self.patchify(x)
            x = self.x_embedder(x)

        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = y.to(x.dtype)
        attn_mask = None
        if mask is not None:
            if mask.ndim == 2:
                attn_mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.ndim == 3:
                attn_mask = mask.unsqueeze(1)
            
            if attn_mask.dtype == torch.bool:
                 new_mask = torch.zeros_like(attn_mask, dtype=x.dtype)
                 new_mask.masked_fill_(~attn_mask, float("-inf"))
                 attn_mask = new_mask
            else:
                 attn_mask = attn_mask.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        if self.model_type == 'decoder':
            x = self.final_layer(x, adaln_input)
            x = self.unpatchify(x)  # (N, out_channels, H, W)

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

def loss_vae(inputs, recon, posterior, mask=None, kl_weight=1e-6, decoder_type="reg", num_bins=512, normal_weight=1e-3):
    # 1. 计算原始 diff
    if decoder_type == "cls":
        target_idx = float_to_index(inputs, num_bins=num_bins).to(inputs.device) # [B, N, 9] long type
        rec_diff = F.cross_entropy(
            recon.permute(0, 3, 1, 2),  # [B, N, 9, C] -> [B, C, N, 9]
            target_idx, 
            reduction='none'
            )   
    elif decoder_type == "reg":
        rec_diff = torch.abs(inputs - recon) # [b, n, 9]
                
    kl_diff = posterior.kl() if posterior is not None else torch.zeros_like(rec_diff)

    rec_loss = _masked_mean(rec_diff, mask)
    kl_loss = _masked_mean(kl_diff, mask)

    # 3. 加权求和
    loss = rec_loss + kl_weight * kl_loss
    
    return loss, rec_loss, kl_loss



if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, c = 800, 9
    model = AutoencoderKL(latent_channels=9).cuda()

    x = torch.randn(2, n, c).cuda() # Batch=2
    cond = torch.randint(0, 16, (2,)).cuda() # Batch=2
    mask = torch.ones(2, n).bool().cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        recon, posterior, z = model(x, cond=cond, mask=mask)

    loss, rec_l, kl_l = loss_vae(x, recon, posterior, mask=mask)
    model.eval()
    
