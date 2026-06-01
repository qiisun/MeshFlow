#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from datasets.mesh_dataset import save_mesh
from flow_matching import create_transport
from inference import (
    _configure_nccl_env_for_consumer_rtx,
    _load_val_face_lengths,
    _make_sample_fn,
    apply_overrides,
    build_model_from_config,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--length_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def selected_step_indices(num_steps, stride):
    stride = max(1, stride)
    indices = list(range(0, num_steps, stride))
    final_index = num_steps - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return indices


def main():
    args = parse_args()
    _configure_nccl_env_for_consumer_rtx()
    torch.manual_seed(args.seed)

    train_config = load_config(args.config)
    overrides = [
        f"ckpt_path={args.ckpt_path}",
        f"sample.num_sampling_steps={args.num_steps}",
    ]
    if args.cfg_scale is not None:
        overrides.append(f"sample.cfg_scale={args.cfg_scale}")
    overrides.extend(args.overrides)
    apply_overrides(train_config, overrides)

    checkpoint_step = Path(args.ckpt_path).stem
    output_dir = args.output or os.path.join(
        train_config["train"]["output_dir"],
        train_config["train"]["exp_name"],
        f"infer_{checkpoint_step}_denoise",
    )
    os.makedirs(output_dir, exist_ok=True)
    if args.clean:
        for mesh_path in Path(output_dir).glob("*.obj"):
            mesh_path.unlink()

    accelerator = Accelerator()
    device = accelerator.device
    model = build_model_from_config(train_config)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["ema"])
    model = model.to(device).eval()
    model_unwrap = accelerator.unwrap_model(model)

    transport = create_transport(
        train_config["transport"]["path_type"],
        train_config["transport"]["prediction"],
        use_lognorm=train_config["transport"].get("use_lognorm", False),
        use_jit=train_config["transport"].get("use_jit", False),
    )
    sample_fn = _make_sample_fn(
        transport,
        train_config["sample"],
        train_config["sample"].get("timestep_shift", 0.0),
    )

    face_lengths = _load_val_face_lengths(train_config)
    if not face_lengths:
        raise ValueError("Validation face lengths unavailable")
    valid_length = int(face_lengths[args.length_index % len(face_lengths)])
    max_length = int(train_config["model"].get("max_length", 800))
    face_lengths_tensor = torch.tensor([valid_length], device=device, dtype=torch.long)
    mask = torch.arange(max_length, device=device).unsqueeze(0) < face_lengths_tensor.unsqueeze(1)

    noise = torch.randn(1, max_length, 9, device=device)
    guided_noise = torch.cat([noise, noise], dim=0)
    cfg_scale = float(train_config["sample"].get("cfg_scale", 2.0))

    with torch.no_grad():
        trajectory = sample_fn(
            guided_noise,
            model_unwrap.forward_with_cfg,
            y=face_lengths_tensor,
            cfg_scale=cfg_scale,
            mask=mask,
        )

    indices = selected_step_indices(int(trajectory.shape[0]), args.stride)
    for frame_index, step_index in enumerate(indices):
        tokens = trajectory[step_index, 0, :valid_length].detach().cpu().numpy()
        save_path = os.path.join(output_dir, f"{frame_index:03d}_{step_index:03d}.obj")
        save_mesh(tokens, save_path, max_val=1 / 0.3762)

    print(f"[DONE] Saved {len(indices)} OBJ frames to {output_dir}")
    print(f"[INFO] valid_length={valid_length}, trajectory_steps={trajectory.shape[0]}, cfg_scale={cfg_scale}")


if __name__ == "__main__":
    main()