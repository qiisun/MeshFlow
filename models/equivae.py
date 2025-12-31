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
                 latent_channels=3,
                 decoder_type="reg",
                 num_bins=256,
                 use_rmsnorm=False,
                 fixed_noise=True,
                 noise_level=None,
                 use_encoder=False,
                 use_coord_encoding_decoder=True,
                 use_equi_model=True):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_bins = num_bins
        self.use_encoder = use_encoder
        self.fixed_noise = fixed_noise
        self.noise_level = noise_level
        self.use_coord_encoding_decoder = use_coord_encoding_decoder
        if self.use_coord_encoding_decoder:
            self.post_quant_linear = XEmbedder(hidden_dim, 15, True, 3, max_range=0.5*1.05)
        else:
            self.post_quant_linear = nn.Linear(latent_channels, hidden_dim*3)  
        
        if self.use_encoder:
            self.quant_linear = nn.Linear(hidden_dim*3, 2*latent_channels) # 9 -> 3*2
            if use_equi_model:
                self.encoder = EquiModel(hidden_dim=hidden_dim, model_type='encoder', num_bins=num_bins, use_rmsnorm=use_rmsnorm) # encoder
            else:
                self.encoder = Model(dim=hidden_dim, model_type='encoder')
        
        if use_equi_model:
            self.decoder = EquiModel(hidden_dim=hidden_dim, model_type='decoder', decoder_type=decoder_type, num_bins=num_bins, use_rmsnorm=use_rmsnorm) # decoder   
        else:
            from models.equivae1 import Model
            self.decoder = Model(dim=hidden_dim, model_type='decoder')

    def encode(self, x, cond, mask):
        """
        Input: [B, N, 9]
        Output Posterior over: 
        [B, 3*N, latent_dim]
        or
        [B, N, 3]
        """        
        if self.use_encoder:
            h = self.encoder(x, cond, mask)
            bs, n_vertices, c = h.shape
            h = h.reshape(bs, n_vertices//3, -1)            
            moments = self.quant_linear(h) # [bs, n, 3]
        else:
            bs, n, _ = x.shape
            moments = x.reshape(bs, 3*n, self.latent_channels)
            mask = mask.repeat_interleave(3, dim=1)
        posterior = DiagonalGaussianDistribution(moments, mask=mask, noise_level=self.noise_level if self.noise_level is not None else None)
        return posterior
    
    def decode(self, z, cond, mask):
        """
        解码过程：输入 Latent -> 重建图像
        """
        if self.use_coord_encoding_decoder:
            bs, n_vertices, _ = z.shape
            z = z.reshape(bs, -1, 9)
            _, freq_z = self.post_quant_linear(z) # [very strange shape]
            freq_z = freq_z.reshape(bs, n_vertices, -1) # [bs, 3*N, c]
            dec = self.decoder(freq_z, cond, mask) # [bs, 3*N, c] -> [bs, N, 9]
            dec = z + dec # residual connection
        else:
            # TODO: without positional encoding
            z = self.post_quant_linear(z) 
            bs, N, _ = z.shape
            z = z.reshape(bs, 3*N, -1)
            dec = self.decoder(z, cond, mask)
        return dec

    def forward(self, input, cond, mask=None, sample_posterior=True):
        """
        端到端的前向传播：包含 Loss 计算
        这是训练时调用的主函数
        """
        # 1. Encode
        posterior = self.encode(input, cond, mask)
        
        # 2. Sample
        if sample_posterior:
            z = posterior.sample() # [b, 3*N, 3] 
        else:
            z = posterior.mode() # z[mask.unsqueeze(2).repeat(1,1,3).reshape(bs, -1)]
        reconstruction = self.decode(z, cond, mask)
        return reconstruction, posterior, z

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, mask=None, deterministic=False, noise_level=None):
        """
        parameters: [B, N, 2*C]
        mask: [B, N] or [B, N, 1] 
        """
        self.parameters = parameters
        if noise_level is not None:
            self.noise_level = noise_level
            self.mean = parameters
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
            self.logvar = torch.clamp(self.logvar, -6.0, 20.0)
            self.deterministic = deterministic
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)
            self.noise_level = None
        self.mask = mask

    def sample(self):
        if self.noise_level is None:
            x = self.mean + self.std * torch.randn_like(self.mean)
            if self.mask is not None:
                if self.mask.dim() == 2:
                    m = self.mask.unsqueeze(-1)
                else:
                    m = self.mask
                x = x * m 
        else:
            x = self.mean + self.noise_level * torch.randn_like(self.mean)
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
        if self.noise_level is None:
            kl_val = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
            if self.mask is not None:
                if self.mask.dim() == 2:
                    m = self.mask.unsqueeze(-1)
                else:
                    m = self.mask
                kl_val = kl_val * m
            return kl_val
        else:
            return 0
       

# modelify from equiDIT
class EquiModel(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, max_length=800,
                 num_layers=6, gradient_checkpointing=True, 
                 class_dropout_prob=0.1, use_coord_encoding_decoder=True, 
                 version=3, pe_freq=20, mixed_precision='bf16',
                 use_dit_like_pe=False, face_cond=True, face_bin=100,
                 use_rmsnorm=True, 
                 model_type='encoder',
                 decoder_type="reg",
                 num_bins=512,
                 zero_init_final=False):
        super().__init__()
        self.model_type = model_type
        self.decoder_type = decoder_type
        self.use_dit_like_pe = use_dit_like_pe
        self.kl_weight = 1e-6
        self.use_l1_loss = True

        input_dim = 3 if version > 1 else 9
        self.version = version
        self.use_coord_encoding_decoder = use_coord_encoding_decoder
        self.hidden_size = hidden_dim
        self.zero_init_final = zero_init_final
        # project input
        if model_type == 'encoder':
            self.x_embedder = XEmbedder(hidden_dim, pe_freq=pe_freq, use_coord_encoding=True, version=version, max_range=0.5)
            
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
        
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.model_type == 'decoder' and self.zero_init_final:
            # Zero-out output layers:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
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
    """
    计算三角形面的法向量
    vertices: [B, N, 9] (每个面3个顶点，平铺)
    return: [B, N, 3] (归一化的法向# The above code is a comment in Python. Comments are used to provide
    # explanations or notes within the code for better understanding. In
    # Python, comments start with the `#` symbol and everything after the `#`
    # on that line is considered a comment and is ignored by the Python
    # interpreter.
    量)
    """
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

def loss_vae(inputs, recon, posterior, mask=None, kl_weight=1e-6, decoder_type="reg", num_bins=512, normal_weight=1e-3,
             fixed_noise=True, loss_type = "mse", noise_scale=None):
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
        mse = (inputs - recon)**2
    mae = _masked_mean(rec_diff, mask)
    # if noise_scale is not None:
    #     mse = _masked_mean(mse, mask) / noise_scale
    # else:
    mse = _masked_mean(mse, mask) 
    
    if fixed_noise:
        if loss_type == 'mae':
            return mae, 0, mae, torch.sqrt(mse)
        elif loss_type == 'mse':
            return mse, 0, mae, torch.sqrt(mse)
    else:
        kl_diff = posterior.kl() # [B, N] or [B, N, C]
        kl_loss = _masked_mean(kl_diff, mask)
        loss = mse + kl_weight * kl_loss
        return loss, kl_loss, mae, torch.sqrt(mse)

if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, c = 800, 9
    model = AutoencoderKL(latent_channels=3,
                          fixed_noise=False,
                          use_encoder=True,
                          use_coord_encoding_decoder=False).cuda()

    x = torch.randn(2, n, c).cuda() # Batch=2
    cond = torch.randint(0, 16, (2,)).cuda() # Batch=2
    mask = torch.ones(2, n).bool().cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        recon, posterior, z = model(x, cond=cond, mask=mask)
    loss, kl_l, mae, rmse = loss_vae(x, recon, posterior, mask=mask,
                                 fixed_noise=False)
    print(loss.item()) # init with 1e-4 if mse
    model.eval()
    
