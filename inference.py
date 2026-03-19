"""
Compatibility wrapper.
Core inference implementation is inference_dit.py.
"""

import argparse
from accelerate import Accelerator

from inference_dit import do_sample, load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_jit.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)

    assert 'ckpt_path' in train_config, "ckpt_path must be specified in config"
    ckpt_dir = train_config['ckpt_path']

    do_sample(train_config, accelerator, ckpt_path=ckpt_dir, demo_sample_mode=args.demo)
