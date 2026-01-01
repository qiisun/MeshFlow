import sys
sys.path.append('.')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.pos_embed import get_embedder
from models.swiglu_ffn import SwiGLUFFN 
from models.rmsnorm import RMSNorm
from models.attention import SelfAttention
from models.utils import get_2d_sincos_pos_embed

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, nonlin_type="swiglu"):
        super().__init__()
        mlp_hidden_dim = int(dim * mult)
        if nonlin_type == "geglu":
            self.net = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim * 2),
                GEGLU(),
                nn.Linear(mlp_hidden_dim, dim)
            )
        elif nonlin_type == "silu":
            self.net = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(mlp_hidden_dim, dim)
            )
        elif nonlin_type == "swiglu":
            self.net = SwiGLUFFN(dim, int(2/3 * mlp_hidden_dim))
        else:
            raise ValueError(f"Invalid nonlin_type: {nonlin_type}")

    def forward(self, x, mask=None, version=3):
        if version == 3:
            if mask is not None:
                B, N, D = x.shape
                # Maks is (B, N), X is (B, N, D)
                x = x.reshape(B, N//3, 3, D)[mask]
            y = self.net(x)
            if mask is not None:
                y_ = torch.zeros(B, N//3, 3, D, device=x.device, dtype=y.dtype)
                y_[mask] = y
                y = y_.reshape(B, N, D)
            return y
        else:
            B, N, D = x.shape
            x = x[mask]
            y = self.net(x)
            y_ = torch.zeros(B, N, D, device=x.device, dtype=y.dtype)
            y_[mask] = y
            y = y_.reshape(B, N, D)
            return y

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class XEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, pe_freq, use_coord_encoding=True, version=2):
        super().__init__()
        self.version = version
        embed_fn, frequency_embedding_size = get_embedder(pe_freq, input_dims=3 if version > 1 else 9)  # 10 -> 20
        self.embed_fn = embed_fn
        self.input_dim = frequency_embedding_size if use_coord_encoding else 9
        self.use_coord_encoding = use_coord_encoding
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.hidden_size = hidden_size

    def x_embedding(self, x, max_period=10.): # [N, 9] -> [N, 180]
        return self.embed_fn(x / max_period + 0.5) # renormalize to [0, 1]
    
    def forward(self, x):
        if self.version > 1:
            b, n, _ = x.shape
            x = x.view(b, n*3, 3)
        if self.use_coord_encoding:
            x = self.x_embedding(x=x)
        x_emb = self.mlp(x) # [bs, N*3, hidden_dim]
        if self.version > 1:
            x_emb = x_emb.view(b, n, 3, self.hidden_size)
            x_emb_2 = x_emb.view(b*n, 3, self.hidden_size) # keep the order 
            x_emb_1 = x_emb.mean(dim=2) # pooled face feature
            return x_emb_1, x_emb_2
        else:
            return x_emb
    

class VertexCrossAttention(nn.Module):
    def __init__(self, modulation_type="add", hidden_dim=1024):
        super().__init__()
        self.modulation_type = modulation_type
        if self.modulation_type == "concat":
            self.linear = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
    
    def forward(self, x, x_):
        # [B, 3, C] x [B, 1, C] -> [B, 3, C]
        if self.modulation_type == "add":
            return  x + x_.repeat(1, 3, 1)
        elif self.modulation_type == "mult":
            return  x * x_.repeat(1, 3, 1)
        elif self.modulation_type == "concat":
            return self.linear(torch.cat([x, x_.repeat(1, 3, 1)], dim=-1))
        else:
            raise ValueError(f"Invalid modulation type: {self.modulation_type}")
        

    
# PixArtAlpha-style
class DiTLayer(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True, mixed_precision='bf16', 
                version=3, 
                nonlin_type="silu", # 'swiglu'
                use_rmsnorm=False,
                mlp_ratio=4.0,
                use_qknorm=False, # used for stablizing training
                ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.version = version

        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        
        else:
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
            
        self.attn = SelfAttention(dim, num_heads, mixed_precision=mixed_precision)
        self.mlp = FeedForward(dim, mult=mlp_ratio, nonlin_type=nonlin_type)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
    def forward(self, x, c, mask=None):
        if self.version <= 2:
            if self.training and self.gradient_checkpointing:
                return checkpoint(self._forward, x, c, mask=mask, use_reentrant=False)
            elif self.version <= 2:
                return self._forward(x, c, mask=mask)
        else:
            if self.training and self.gradient_checkpointing:
                return checkpoint(self._forward_v3, x, c, mask=mask, use_reentrant=False)
            else:
                return self._forward_v3(x, c, mask=mask)
    
    def _forward(self, x, c, mask=None):
        # x: [B, N, C], hidden states
        # t_adaln: [B, 6, C], timestep embedding of adaln
        # return: [B, N, C], updated hidden states
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask=mask) # masking padded tokens to avoid the attention to see the future
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp),
                                                mask=mask, version =self.version)
        return x
    
    def _forward_v3(self, x, c, mask=None):
        # x: [B, N*3, C], hidden states
        # t_adaln: [B, 6, C], timestep embedding of adaln
        # return: [B, N, 3, C], updated hidden states
        b, n_vertex, _ = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)                
        x_face = x.reshape(b, n_vertex//3, 3, -1).mean(dim=2) # [B, n_face, C]
        x_face = gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x_face), shift_msa, scale_msa), mask=mask) 
        x = x + x_face.unsqueeze(2).repeat(1,1,3,1).reshape(b, n_vertex, self.dim) # [B, n_face, 3, C]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp),
                                                mask=mask) # [B, n_vertex, C]
        return x

class DiT(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, max_length=800, 
                num_layers=24, gradient_checkpointing=True, 
                class_dropout_prob=0.1, use_coord_encoding=True, 
                version=2, pe_freq=20, mixed_precision='bf16',
                use_dit_like_pe=False, face_cond=False, face_bin=None,
                use_rmsnorm=False,
                use_repa=False,
                is_latent=False
                ):
        super().__init__()
        input_dim = 3 if version > 1 else 9
        self.input_dim = input_dim
        self.version = version
        self.use_coord_encoding = use_coord_encoding
        self.hidden_size = hidden_dim
        self.use_repa = use_repa
        self.is_latent = is_latent
        self.use_dit_like_pe = use_dit_like_pe
        
        # project input
        self.x_embedder = XEmbedder(hidden_dim, pe_freq=pe_freq, use_coord_encoding=False if is_latent else True, version=version)
        
        # positional encoding (just use a learnable positional encoding)
        if use_dit_like_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, max_length, hidden_dim) / hidden_dim ** 0.5)
            print("[WARNING] USE DiT-like POSITIONAL ENCODING")
        else: # default
            print("[INFO] DO NOT USE POSITIONAL ENCODING")
        
        # timestep encoding
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.face_cond = face_cond
        if face_cond:
            self.uncond_y = max_length + 1
            self.num_classes = max_length//face_bin
            self.y_embedder = LabelEmbedder(max_length//face_bin, hidden_dim, class_dropout_prob) # 16+1
            self.face_bin = face_bin

        # transformer layers
        self.layers = nn.ModuleList([DiTLayer(hidden_dim, num_heads, gradient_checkpointing, mixed_precision=mixed_precision, version=version, use_rmsnorm=use_rmsnorm) for _ in range(num_layers)])

        # project out     
        self.final_layer = FinalLayer(hidden_dim, input_dim, use_rmsnorm=use_rmsnorm) # input_dim: 9 for v2, 3 for v3
        
        if self.use_repa:
            self.proj = nn.Sequential(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 448),
                )
            )

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

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.x_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.x_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(self, x, t, y, mask=None):
        # x: [B, N, C], hidden states
        # c: [B, M, C]
        # y: [B,], face number
        # t: [B,], timestep, float from [0,1]
        # mask: [B, N]
        # return: [B, N, C], updated hidden states
        B, N, _ = x.shape
        
        if self.version > 1:
            _, x = self.x_embedder(x)
            x = x.view(B, N*3, -1) # [B, 3*N, C]
        else:
            x = self.x_embedder(x) # [B, N, C]
            
        # positional encoding
        if self.use_dit_like_pe: 
            if self.version > 1:
                x = x + self.pos_embed[:, :N*3, :]
            else:
                x = x + self.pos_embed[:, :N, :]

        # only timestep encoding
        t = self.t_embedder(t)
        if self.face_cond:
            y = self.y_embedder(y // self.face_bin, self.training)
            c = t + y
        else:
            c = t 

        # transformer layers
        # if self.version > 2: # v3
        #     x = x.reshape(B, N*3, -1) # [B, N*3, C]
        # else:
        #     x = x
            
        for layer in self.layers:
            x = layer(x, c, mask=mask)
        
        # project out            
        x = self.final_layer(x, c) # v2: [B, N*3, 3]
        
        if self.version > 1:
            x = x.view(B, N, 9)
        return x
    
    def forward_with_cfg(self, x, t, y, mask=None, cfg_scale=1.0):
        # 1. 确保输入 latent (x) 的条件和无条件分支完全一致
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        # 2. 处理 Mask：使其与 combined 的结构对齐
        combined_mask = None
        if mask is not None:
            mask_half = mask[: len(mask) // 2]
            combined_mask = torch.cat([mask_half, mask_half], dim=0)

        # 3. 前向传播
        model_out = self.forward(combined, t, y, mask=combined_mask)
        
        eps, rest = model_out[:, :self.input_dim], model_out[:, self.input_dim:]
        
        # 5. 计算 CFG
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        return torch.cat([eps, rest], dim=1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels, input_size=None, use_rmsnorm=False):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(input_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(input_size)
        self.linear = nn.Linear(input_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * input_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    from kiui.nn.utils import count_parameters
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version = 3
    model = DiT(version=version).to(device)
    model.eval()
    
    total, trainable = count_parameters(model)
    print(f'[INFO] param total: {total/1024**2:.2f}M, trainable: {trainable/1024**2:.2f}M')

    # test forward
    x = torch.randn(4, 800, 9, device=device, dtype=torch.bfloat16)
    t = torch.randint(0, 1000, (4,), device=device)
    c = torch.randint(0, 55, (4,), device=device) # [B,]
    mask = torch.ones_like(x[:, :, 0], device=device, dtype=torch.bool) # [B, N]
    x_attn = torch.randn(4, 800, 1024, device=device, dtype=torch.bfloat16)
    
    model_attneion = SelfAttention(1024, 16, 1024, 1024, dropout=0.0).to(device)
    
    def model_foward(x):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            y = model(x, t, c, None)
        return y
    
    model_foward(x)

    def test_eq_attn(x):
        # 只permute face
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            y = model_attneion(x_attn)
        
        perm_face = torch.randperm(x.shape[1])
        x_perm_face = x_attn[:, perm_face]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            y_perm_face = model_attneion(x_perm_face)
        print(y_perm_face[0,0,], y[:, perm_face][0,0])
        # print(torch.norm(y_perm_face-y).item() / torch.norm(y_perm_face).item() )
        print(torch.max(torch.abs(y_perm_face-y[:, perm_face]) / (torch.abs(y_perm_face)+ 1e-6)))

    def test_eq_face(x):
        # 只permute face
        y = model_foward(x)
        perm_face = torch.randperm(x.shape[1])
        x_perm_face = x[:, perm_face]
        y_perm_face = model_foward(x_perm_face)
        perm_y = y[:, perm_face]
        print(y_perm_face[0,0,], y[:, perm_face][0,0])
        print(torch.max(torch.abs(y_perm_face - y) / (torch.abs(y_perm_face)+ 1e-6)))
        print(torch.max(torch.abs(y_perm_face - perm_y) / (torch.abs(y_perm_face)+ 1e-6)))
        print(torch.mean(torch.abs(y_perm_face - y) / (torch.abs(y_perm_face)+ 1e-6)))
        print(torch.mean(torch.abs(y_perm_face - perm_y) / (torch.abs(y_perm_face)+ 1e-6)))


    def test_eq_vertex(x):
        # permute face和vertex
        y = model_foward(x)
        x = x.view(x.shape[0], x.shape[1], 3, -1)
        perm_vertex = torch.randperm(x.shape[2])
        x_perm = x[:,:, perm_vertex].view(x.shape[0], x.shape[1], -1)
        y_perm = model_foward(x_perm) # [BS, N, 9]
        
        # prepare the permuted y
        perm_y = y.view(x.shape[0], x.shape[1], 3, -1)
        perm_y = perm_y[:,:, perm_vertex].view(x.shape[0], x.shape[1], -1) # [BS, N, 9]
        
        # check
        print(y_perm[0,0], perm_y[0,0])
        print(torch.max(torch.abs(y_perm - y) / (torch.abs(y_perm)+ 1e-6)))
        print(torch.max(torch.abs(y_perm - perm_y) / (torch.abs(y_perm)+ 1e-6)))
        print(torch.mean(torch.abs(y_perm - y) / (torch.abs(y_perm)+ 1e-6)))
        print(torch.mean(torch.abs(y_perm - perm_y) / (torch.abs(y_perm)+ 1e-6)))
        
        
    def test_eq_random_shuffle(x):
        # permute 不同的face的vertex 
        y = model_foward(x)
        perm_face, perm_vertex = torch.randperm(x.shape[1]), torch.randperm(x.shape[2])
        x_perm_face = x[:, perm_face][:,:, perm_vertex]
        y_perm_face = model_foward(x_perm_face)
        print(torch.allclose(y_perm_face, y[:, perm_face][:, :, perm_vertex]))
    
    # test_eq_face(x)
    # test_eq_attn(x_attn)
    # test_eq_vertex(x)