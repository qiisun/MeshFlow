'''
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import torch
import numpy as np
import trimesh
import megfile
import sys
sys.path.append('.')
# from core.options import Options
import logging

def init_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # write to file
    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # print to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

def load_mesh(path):
    # path: string for local/s3 path
    if path.startswith('s3'):
        ext = path.split('.')[-1]
        with megfile.smart_open(path, 'rb') as f:
            _data = trimesh.load(file_obj=trimesh.util.wrap_as_stream(f.read()), file_type=ext)
    else:
        _data = trimesh.load(path)

    # always convert scene to mesh, and apply all transforms...
    if isinstance(_data, trimesh.Scene):
        # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
        _concat = []
        # loop the scene graph and apply transform to each mesh
        scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
        for k, v in scene_graph.items():
            name = v['geometry']
            if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                transform = v['transform']
                _concat.append(_data.geometry[name].apply_transform(transform))
        _mesh = trimesh.util.concatenate(_concat)
    else:
        _mesh = _data
    
    vertices = _mesh.vertices
    faces = _mesh.faces

    return vertices, faces


def normalize_mesh(vertices, bound=0.95):
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices


# def get_tokenizer(opt: Options):
#     if opt.use_meto:
#         from meto import Engine
#         tokenizer = Engine(discrete_bins=opt.discrete_bins, backend=opt.meto_backend)
#         vocab_size = tokenizer.num_tokens + 3
#     else:
#         tokenizer = None
#         vocab_size = opt.discrete_bins + 3
#     return tokenizer, vocab_size


def quantize_num_faces(n):
    # 0: <=0, un cond
    # 1: 0-1000, low-poly
    # 2: 1000-2000, mid-poly
    # 3: 2000-4000, high-poly
    # 4: 4000-8000, ultra-poly
    if isinstance(n, int):
        if n <= 0:
            return 0
        elif n <= 1000:
            return 1
        elif n <= 2000:
            return 2
        elif n <= 4000:
            return 3
        elif n <= 8000:
            return 4
        else:
            return 5
    else: # torch tensor
        results = torch.zeros_like(n)
        # results[n <= 0] = 0
        results[(n > 0) & (n <= 1000)] = 1
        results[(n > 1000) & (n <= 2000)] = 2
        results[(n > 2000) & (n <= 4000)] = 3
        results[(n > 4000) & (n <= 8000)] = 4
        results[n > 8000] = 5
        return results
    
def monkey_patch_transformers():
    import torch
    import math
    from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, ExponentialDecayLengthPenalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        # MODIFICATION: use input_ids.shape[0] instead of -1 to avoid confusion
        for batch_id, beam_sent in enumerate(input_ids.view(input_ids.shape[0], self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed
    
    PrefixConstrainedLogitsProcessor.__call__ = __call__
    print(f'[INFO] monkey patched PrefixConstrainedLogitsProcessor.__call__')


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
"""This is an ad-hoc sampling schedule that was proposed in https://arxiv.org/abs/2206.00364 it works very well for cifar 10 so we added its implementation here. It did not yield an improvement on ImageNet."""


def get_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples

def plot_face_loss(pred, mask, face_prop):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(face_prop, pred[mask].detach().cpu().numpy(), 'o', label='pred')
    plt.plot(face_prop, mask.float().detach().cpu().numpy(), 'o', label='mask')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    time_grid = get_time_discretization(nfes=50)
    print(time_grid)