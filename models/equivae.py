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
                 use_rmsnorm=False):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_bins = num_bins
        self.quant_linear = nn.Linear(hidden_dim, 2 * latent_channels) # 9 -> 4*2

        self.post_quant_linear = nn.Linear(latent_channels, hidden_dim)   
        
        self.encoder = Model(hidden_dim=hidden_dim, model_type='encoder', num_bins=num_bins, use_rmsnorm=use_rmsnorm) # encoder 
        self.decoder = Model(hidden_dim=hidden_dim, model_type='decoder', decoder_type=decoder_type, num_bins=num_bins, use_rmsnorm=use_rmsnorm) # decoder   
        
    def encode(self, x, cond, mask):
        """
        Input: [B, N, 9]
        Output Posterior over: [B, 3*N, latent_dim]
        """
        h = self.encoder(x, cond, mask)
        import ipdb; ipdb.set_trace()
        moments = self.quant_linear(h)
        mask_expanded = mask.repeat_interleave(3, dim=1)

        posterior = DiagonalGaussianDistribution(moments, mask=mask_expanded)
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
            z = posterior.mode()
            
        reconstruction = self.decode(z, cond, mask)
        return reconstruction, posterior, z

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, mask=None, deterministic=False):
        """
        parameters: [B, N, 2*C]
        mask: [B, N] or [B, N, 1] 
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
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
                 use_dit_like_pe=False, face_cond=True, face_bin=10,
                 use_rmsnorm=False, 
                 model_type='encoder',
                 decoder_type="reg",
                 num_bins=512):
        super().__init__()
        self.model_type = model_type
        self.decoder_type = decoder_type
        
        self.kl_weight = 1e-6
        self.use_l1_loss = True

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
            self.use_dit_like_pe = True
            print("[WARNING] USE DiT-like POSITIONAL ENCODING")
        else: # default
            print("[INFO] DO NOT USE POSITIONAL ENCODING")
            self.use_dit_like_pe = False
        
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
    return: [B, N, 3] (归一化的法向量)
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
        # pred_normals = compute_face_normals(recon)   # [B, N, 3]
        # gt_normals = compute_face_normals(inputs)    # [B, N, 3]
        # cosine_sim = F.cosine_similarity(pred_normals, gt_normals, dim=-1)
        # normal_diff = 1.0 - cosine_sim # [B, N]
        # normal_loss = _masked_mean(normal_diff, mask)
        # rec_diff += normal_loss * normal_weight
        
    kl_diff = posterior.kl() # [B, N] or [B, N, C]

    rec_loss = _masked_mean(rec_diff, mask)
    kl_loss = _masked_mean(kl_diff, mask)

    # 3. 加权求和
    loss = rec_loss + kl_weight * kl_loss
    
    return loss, rec_loss, kl_loss



if __name__ == '__main__':    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, c = 800, 9
    model = AutoencoderKL(latent_channels=4).cuda()

    x = torch.randn(2, n, c).cuda() # Batch=2
    cond = torch.randint(0, 16, (2,)).cuda() # Batch=2
    mask = torch.ones(2, n).bool().cuda()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        recon, posterior, z = model(x, cond=cond, mask=mask)

    loss, rec_l, kl_l = loss_vae(x, recon, posterior, mask=mask)
    model.eval()
    
