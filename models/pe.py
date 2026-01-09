import torch
import torch.nn as nn
import numpy as np

class NeRFEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        self.output_dim = (3 * (2 * num_freqs + (1 if include_input else 0)))

    def forward(self, x):
        embed_fns = []
        
        if self.include_input:
            embed_fns.append(x)
            
        for freq in self.freq_bands:
            embed_fns.append(torch.sin(x * freq * np.pi))
            embed_fns.append(torch.cos(x * freq * np.pi))
        return torch.cat(embed_fns, dim=-1)
    
if __name__ == "__main__":
    x = torch.randn(10, 21, 3)
    encoder = NeRFEncoding(num_freqs=10) # L=10
    encoded_x = encoder(x)

    print(f"输入维度: {x.shape}")        # [1, 3]
    print(f"输出维度: {encoded_x.shape}") # [1, 63] (3 * (2*10) + 3)
    print(encoder.output_dim)
