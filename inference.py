"""
Compatibility wrapper.
Core inference implementation is inference_dit.py.
"""

import argparse
from accelerate import Accelerator

from inference_dit import do_sample_simple, do_sample, load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_jit.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)

    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    ckpt_dir = train_config['ckpt_path']

    from models.dit import DiT_Llama

    model = DiT_Llama(
        in_channels=train_config['model'].get('in_channels', 9),
        input_size=train_config['model'].get('input_size', 32),
        patch_size=train_config['model'].get('patch_size', 1),
        dim=train_config['model'].get('hidden_dim', 1024),
        n_layers=train_config['model'].get('num_layers', 24),
        n_heads=train_config['model'].get('num_heads', 16),
        multiple_of=train_config['model'].get('multiple_of', 256),
        ffn_dim_multiplier=train_config['model'].get('ffn_dim_multiplier', None),
        norm_eps=train_config['model'].get('norm_eps', 1e-5),
        class_dropout_prob=train_config['model'].get('class_dropout_prob', 0.1),
        num_classes=train_config['model'].get('num_classes', 1000),
        face_cond=train_config['model'].get('face_cond', False),
        face_bin=train_config['model'].get('face_bin', 10),
        max_length=train_config['model'].get('max_length', 800),
        use_nerf_pe=train_config['model'].get('use_nerf_pe', train_config['model'].get('use_coord_encoding', True)),
        nerf_num_freqs=train_config['model'].get('nerf_num_freqs', 12),
        nerf_input_range=train_config['model'].get('nerf_input_range', 3.0),
    )

    do_sample(train_config, accelerator, ckpt_path=ckpt_dir, model=model, demo_sample_mode=args.demo)
