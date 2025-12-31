import sys
sys.path.append('.')
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from models.dit import TransformerBlock, LabelEmbedder

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        # )
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, c):
        x = self.linear(x)
        return x


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

# modelify from equiDIT
class Model(nn.Module):
    def __init__(
        self,
        in_channels=9,
        input_size=32,
        patch_size=1,
        dim=768,
        n_layers=12,
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
            self.linear_layer = nn.Linear(dim * 3, dim) 

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
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
                
        for block in self.layers:
            if hasattr(block, 'adaLN_modulation') and len(block.adaLN_modulation) > 0:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        if self.model_type == 'decoder' and hasattr(self, 'final_layer'):
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)
            if hasattr(self.final_layer, 'adaLN_modulation') and len(self.final_layer.adaLN_modulation) > 0:
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def patchify(self, x):
        return x

    def forward(self, x, y=None, mask=None):
        if self.model_type == 'decoder':
            bs, n_vertices, _ = x.shape
            x = x.reshape(bs, n_vertices//3, -1)
            x = self.linear_layer(x) # [bs, n_faces, dim]
        
        self.freqs_cis = self.freqs_cis.to(x.device)
        if self.model_type == 'encoder':
            x = self.init_conv_seq(x)
            x = self.patchify(x)
            x = self.x_embedder(x)

        y = self.y_embedder(y//100, self.training)  # (N, D)
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
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input, mask=attn_mask)

        if self.model_type == 'decoder':
            x = self.final_layer(x, y)
        return x
