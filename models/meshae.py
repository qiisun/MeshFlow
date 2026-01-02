from functools import partial
from math import pi
import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.cuda.amp import autocast
from torchtyping import TensorType
from pytorch_custom_utils import save_load
from beartype import beartype
from beartype.typing import Tuple, Optional
from einops import rearrange, repeat, reduce, pack
from einops.layers.torch import Rearrange
from x_transformers import Encoder
from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.meshgpt_pytorch import (
    discretize, get_derived_face_features,
    scatter_mean, default, pad_at_dim, exists, gaussian_blur_1d,
    undiscretize, first,
)
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F

@save_load()
class MeshVAE(Module):
    @beartype
    def __init__(
        self,
        encoder_depth = 12,
        encoder_heads = 8,
        encoder_dim = 512,
        decoder_fine_dim = 192,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        dim_coor_embed = 64,
        num_discrete_area = 128,
        dim_area_embed = 16,
        num_discrete_normals = 128,
        dim_normal_embed = 64,
        num_discrete_angle = 128,
        dim_angle_embed = 16,
        dim_latent = 192,             # 这里的 latent dimension 对应 continuous latent z 的维度
        kl_loss_weight = 0.0001,      # KL Loss 的权重，通常需要很小
        bin_smooth_blur_sigma = 0.4,
        pad_id = -1,
        patch_size = 1,
        dropout = 0.,
        use_abs_pos = False,
        graph_layers = 1,
    ):
        super().__init__()

        # --- 1. Coordinate & Feature Embedding (保持不变) ---
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete = num_discrete_area, continuous_range = (0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete = num_discrete_normals, continuous_range = coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        init_dim = dim_coor_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed
        
        # --- 2. Patchify & Encoder (保持不变) ---
        self.patch_size = patch_size
        self.project_in = nn.Linear(init_dim * patch_size, encoder_dim)
        
        # Graph init
        self.graph_layers = graph_layers
        if self.graph_layers > 0:
            sageconv_kwargs = dict(normalize = True, project = True)
            self.init_sage_conv = SAGEConv(encoder_dim, encoder_dim, **sageconv_kwargs)
            self.init_encoder_act_and_norm = nn.Sequential(nn.SiLU(), nn.LayerNorm(encoder_dim))
            self.graph_encoders = nn.ModuleList([
                SAGEConv(encoder_dim, encoder_dim, **sageconv_kwargs) for _ in range(graph_layers - 1)
            ])
        
        self.encoder = Encoder(
            dim = encoder_dim,
            depth = encoder_depth,
            heads = encoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )

        # --- 3. VAE Bottleneck (修改部分) ---
        self.dim_latent = dim_latent
        self.to_latent_dist = nn.Linear(encoder_dim, dim_latent * 2)
        
        self.pad_id = pad_id

        # --- 4. Decoder (修改输入层) ---
        
        # 从 continuous latent z 映射回 encoder_dim
        self.init_decoder = nn.Sequential(
            nn.Linear(dim_latent, encoder_dim),
            nn.SiLU(),
            nn.LayerNorm(encoder_dim),
        )

        self.decoder_coarse = Encoder(
            dim = encoder_dim,
            depth = encoder_depth // 2,
            heads = encoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )
        
        self.coarse_to_fine = nn.Linear(encoder_dim, decoder_fine_dim * 3)

        self.decoder_fine = Encoder(
            dim = decoder_fine_dim,
            depth = encoder_depth // 2,
            heads = encoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )

        self.to_coor_logits = nn.Sequential(
            nn.Linear(decoder_fine_dim, patch_size * num_discrete_coors * 3),
            Rearrange('b (nf nv) (v c) -> b nf (nv v) c', nv = 3, v = 3)
        )

        # --- 5. Loss Weights ---
        self.kl_loss_weight = kl_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 3, int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask:        TensorType['b', 'nf', bool],
        face_edges_mask:  TensorType['b', 'e', bool],
        return_face_coordinates = False
    ):
        # ... (这部分代码与原 MeshAutoencoder 完全一致，用于提取特征) ...
        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device
        _, num_faces, _ = faces.shape

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)
        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)
        face_coords = vertices.gather(-2, faces_vertices)

        derived_features = get_derived_face_features(face_coords)

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        discrete_face_coords = self.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') 
        face_coor_embed = self.coor_embed(discrete_face_coords)
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)')

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        
        face_embed = rearrange(face_embed, 'b (num_patch patch_size) d -> b num_patch (patch_size d)', 
                               patch_size = self.patch_size)
        face_embed = self.project_in(face_embed)
        
        face_mask_patch = rearrange(face_mask, 'b (num_patch patch_size) -> b num_patch patch_size', 
                              patch_size = self.patch_size).float().mean(dim=-1).bool()

        # Graph Layers
        if self.graph_layers > 0:
            orig_face_embed_shape = face_embed.shape[:2]
            face_embed_masked = face_embed[face_mask_patch]
            
            # (简化的 Graph 处理逻辑，保持原样)
            face_index_offsets = reduce(face_mask_patch.long(), 'b nf -> b', 'sum')
            face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
            face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')
            
            # 注意：这里的 face_edges 处理可能需要根据 patch logic 调整，这里假设 patch_size=1
            # 如果 patch_size > 1，graph logic 需要对应修改。这里暂且保留原逻辑。
            current_face_edges = face_edges + face_index_offsets
            current_face_edges = current_face_edges[face_edges_mask]
            current_face_edges = rearrange(current_face_edges, 'be ij -> ij be')
            
            face_embed_masked = self.init_sage_conv(face_embed_masked, current_face_edges)
            face_embed_masked = self.init_encoder_act_and_norm(face_embed_masked)

            for conv in self.graph_encoders:
                face_embed_masked = conv(face_embed_masked, current_face_edges)
            
            shape = (*orig_face_embed_shape, face_embed_masked.shape[-1])
            face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask_patch, '... -> ... 1'), face_embed_masked)  
        
        # Transformer Encoder
        face_embed = self.encoder(face_embed, mask=face_mask_patch)
        
        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords, face_mask_patch

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @beartype
    def decode(
        self,
        z: TensorType['b', 'n', 'd', float],
        face_mask: TensorType['b', 'n', bool]
    ):
        # z is continuous latent now
        conv_face_mask = rearrange(face_mask, 'b n -> b n 1')
        vertice_mask = repeat(face_mask, 'b nf -> b (nf nv)', nv = 3)
        
        x = z.masked_fill(~conv_face_mask, 0.)

        x = self.init_decoder(x) # Project back to encoder dim
        x = self.decoder_coarse(x, mask = face_mask)
        x = self.coarse_to_fine(x)
        x = rearrange(x, 'b nf (nv d) -> b (nf nv) d', nv=3)
        x = self.decoder_fine(x, mask = vertice_mask)
        
        return x

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 3, int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        return_loss_breakdown = False,
        return_recon_faces = False,
        only_return_recon_faces = False,
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')
        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')

        # 1. Encode
        encoded, face_coordinates, face_mask_patch = self.encode(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True
        )

        # 2. VAE Bottleneck (Mu, LogVar -> Sample Z)
        # encoded shape: [b, num_patches, encoder_dim]
        # output shape: [b, num_patches, 2 * dim_latent]
        dist_params = self.to_latent_dist(encoded)
        mu, logvar = dist_params.chunk(2, dim=-1)
        
        # Sample z
        z = self.reparameterize(mu, logvar)

        # 3. Decode
        decode_out = self.decode(
            z,
            face_mask = face_mask_patch
        )

        pred_face_coords = self.to_coor_logits(decode_out)

        # 4. Reconstruction output logic
        if return_recon_faces or only_return_recon_faces:
            recon_faces = undiscretize(
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )
            recon_faces = rearrange(recon_faces, 'b nf (nv c) -> b nf nv c', nv = 3)
            face_mask_expanded = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask_expanded, float('nan'))

        if only_return_recon_faces:
            return recon_faces, pred_face_coords, face_coordinates, face_mask

        # 5. Loss Calculation
        
        # A. Reconstruction Loss (Cross Entropy)
        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        with autocast(enabled = False):
            pred_log_prob = pred_face_coords.log_softmax(dim = 1)
            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma)

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1)
            
            # Apply mask
            recon_mask = repeat(face_mask, 'b nf -> b (nf r)', r = 9)
            recon_loss = recon_losses[recon_mask].mean()

        # B. KL Divergence Loss
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        # Mask KL loss by face mask
        kl_loss = kl_loss[face_mask_patch].mean()

        # Total Loss
        total_loss = recon_loss + (kl_loss * self.kl_loss_weight)

        loss_breakdown = (recon_loss, kl_loss)

        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss
            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown

if __name__ == '__main__':
    # Init VAE model
    meshVAE = MeshVAE(
        encoder_dim=256,
        dim_latent=128,  # Continuous latent dimension
        kl_loss_weight=0.001
    )
    
    # Mock input
    vertices = torch.randn(4, 99, 3)
    faces = torch.arange(99).reshape(33, 3)
    faces = repeat(faces, 'f c -> b f c', b=4)
    
    loss, breakdown = meshVAE(vertices=vertices, faces=faces, return_loss_breakdown=True)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Recon Loss: {breakdown[0].item():.4f}, KL Loss: {breakdown[1].item():.4f}")