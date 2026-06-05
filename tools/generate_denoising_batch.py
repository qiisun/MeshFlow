#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch

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


DEFAULT_MAX_VAL = 1.0 / 0.3762
DEFAULT_CLASS_CONFIGS = {
    "lamp": "configs/snet/base-120m-ot-v-lamp.yaml",
    "desk": "configs/snet/base-120m-ot-v-table.yaml",
    "table": "configs/snet/base-120m-ot-v-table.yaml",
}


def natural_key(path):
    name = os.path.basename(str(path))
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", name)]


def parse_key_value(values):
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"expected name=path, got: {value}")
        key, item = value.split("=", 1)
        parsed[key.strip()] = item.strip()
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate batched denoising OBJ trajectories.")
    parser.add_argument("--classes", nargs="+", default=["lamp", "desk", "table"])
    parser.add_argument("--class_config", action="append", default=None, help="Override class config as name=path")
    parser.add_argument("--ckpt", action="append", default=None, help="Override class checkpoint as name=path")
    parser.add_argument("--output_root", default="output/denoise10_lamp_desk_table_100")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--face_count", type=int, default=0, help="Fixed face count; 0 samples validation face counts")
    parser.add_argument("--base_seed", type=int, default=20260605)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing complete sample directories")
    parser.add_argument("--clean_output", action="store_true", help="Delete output_root before generating")
    parser.add_argument("overrides", nargs="*", help="Additional config overrides, e.g. sample.timestep_shift=1")
    return parser.parse_args()


def resolve_device(preference):
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latest_checkpoint_for_config(config):
    train_cfg = config.get("train", {})
    output_dir = train_cfg.get("output_dir", "output")
    exp_name = train_cfg.get("exp_name")
    if not exp_name:
        raise ValueError("config is missing train.exp_name")

    checkpoint_dir = Path(output_dir) / exp_name / "checkpoints"
    checkpoints = sorted(glob.glob(str(checkpoint_dir / "*.pt")), key=natural_key)
    if checkpoints:
        return checkpoints[-1]

    ckpt_path = config.get("ckpt_path")
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path
    raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")


def safe_face_upper(config):
    max_length = int(config["model"].get("max_length", 800))
    face_bin = int(config["model"].get("face_bin", 1) or 1)
    num_classes = max_length // face_bin if face_bin > 0 else max_length
    if num_classes > 0:
        return max(1, min(max_length, num_classes * face_bin) - 1)
    return max(1, max_length - 1)


def choose_face_counts(config, num_samples, requested_face_count, seed):
    safe_upper = safe_face_upper(config)
    if requested_face_count > 0:
        value = int(max(1, min(requested_face_count, safe_upper)))
        return np.full((num_samples,), value, dtype=np.int64)

    face_lengths = [int(x) for x in _load_val_face_lengths(config) if 1 <= int(x) <= safe_upper]
    if face_lengths:
        rng = np.random.default_rng(seed)
        return rng.choice(face_lengths, size=num_samples, replace=True).astype(np.int64)

    fallback = max(32, min(safe_upper, int(config["model"].get("max_length", 800)) // 2))
    return np.full((num_samples,), fallback, dtype=np.int64)


def sample_complete(sample_dir, num_steps):
    if not sample_dir.exists():
        return False
    return len(list(sample_dir.glob("*.obj"))) >= num_steps


def make_noise(sample_ids, class_seed, max_length, device):
    chunks = []
    for sample_id in sample_ids:
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(class_seed + int(sample_id))
        chunks.append(torch.randn(1, max_length, 9, device=device, generator=generator))
    return torch.cat(chunks, dim=0)


def save_trajectory(trajectory, sample_ids, face_counts, class_dir, num_steps):
    for batch_offset, sample_id in enumerate(sample_ids):
        sample_dir = class_dir / f"{sample_id:03d}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        valid_length = int(face_counts[sample_id])
        for step_index in range(num_steps):
            tokens = trajectory[step_index, batch_offset, :valid_length].detach().float().cpu().numpy()
            save_path = sample_dir / f"step_{step_index:03d}.obj"
            save_mesh(tokens, str(save_path), clean=False, max_val=DEFAULT_MAX_VAL)


def generate_class(class_name, config_path, ckpt_path, args, class_index, output_root, manifest_rows):
    class_seed = int(args.base_seed + class_index * 100000)
    random.seed(class_seed)
    np.random.seed(class_seed)
    torch.manual_seed(class_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(class_seed)

    config = load_config(config_path)
    overrides = [
        f"ckpt_path={ckpt_path}",
        f"sample.num_sampling_steps={args.num_steps}",
    ]
    if args.cfg_scale is not None:
        overrides.append(f"sample.cfg_scale={args.cfg_scale}")
    overrides.extend(args.overrides)
    apply_overrides(config, overrides)

    device = resolve_device(args.device)
    model = build_model_from_config(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["ema"])
    model = model.to(device).eval()

    transport = create_transport(
        config["transport"]["path_type"],
        config["transport"]["prediction"],
        use_lognorm=config["transport"].get("use_lognorm", False),
        use_jit=config["transport"].get("use_jit", False),
    )
    sample_fn = _make_sample_fn(transport, config["sample"], config["sample"].get("timestep_shift", 0.0))

    max_length = int(config["model"].get("max_length", 800))
    face_counts = choose_face_counts(config, args.num_samples, args.face_count, class_seed)
    class_dir = output_root / "trajectories" / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = []
    for sample_id in range(args.num_samples):
        sample_dir = class_dir / f"{sample_id:03d}"
        if args.overwrite or not sample_complete(sample_dir, args.num_steps):
            sample_ids.append(sample_id)
        manifest_rows.append(
            {
                "class": class_name,
                "sample_id": f"{sample_id:03d}",
                "face_count": int(face_counts[sample_id]),
                "config": config_path,
                "ckpt": ckpt_path,
                "trajectory_dir": str(sample_dir),
            }
        )

    print(f"[INFO] {class_name}: config={config_path}")
    print(f"[INFO] {class_name}: ckpt={ckpt_path}")
    print(f"[INFO] {class_name}: {len(sample_ids)}/{args.num_samples} samples need generation")
    if not sample_ids:
        return

    cfg_scale = float(config.get("sample", {}).get("cfg_scale", 2.0))
    arange = torch.arange(max_length, device=device).unsqueeze(0)
    batch_size = max(1, int(args.batch_size))

    with torch.no_grad():
        for start in range(0, len(sample_ids), batch_size):
            batch_ids = sample_ids[start : start + batch_size]
            batch_faces = torch.tensor([int(face_counts[i]) for i in batch_ids], device=device, dtype=torch.long)
            mask = arange < batch_faces.unsqueeze(1)
            noise = make_noise(batch_ids, class_seed, max_length, device)
            guided_noise = torch.cat([noise, noise], dim=0)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    trajectory = sample_fn(guided_noise, model.forward_with_cfg, y=batch_faces, cfg_scale=cfg_scale, mask=mask)
            else:
                trajectory = sample_fn(guided_noise, model.forward_with_cfg, y=batch_faces, cfg_scale=cfg_scale, mask=mask)

            if int(trajectory.shape[0]) != args.num_steps:
                raise RuntimeError(f"expected {args.num_steps} trajectory steps, got {trajectory.shape[0]}")

            trajectory = trajectory[:, : len(batch_ids)]
            save_trajectory(trajectory, batch_ids, face_counts, class_dir, args.num_steps)
            print(f"[INFO] {class_name}: saved samples {batch_ids[0]:03d}-{batch_ids[-1]:03d}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def write_manifest(output_root, rows):
    manifest_path = output_root / "trajectories_manifest.csv"
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["class", "sample_id", "face_count", "config", "ckpt", "trajectory_dir"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main():
    args = parse_args()
    _configure_nccl_env_for_consumer_rtx()

    output_root = Path(args.output_root).resolve()
    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_configs = dict(DEFAULT_CLASS_CONFIGS)
    class_configs.update(parse_key_value(args.class_config))
    ckpt_overrides = parse_key_value(args.ckpt)

    resolved = []
    for class_name in args.classes:
        if class_name not in class_configs:
            raise ValueError(f"no config known for class '{class_name}'. Use --class_config {class_name}=path")
        config_path = class_configs[class_name]
        config = load_config(config_path)
        ckpt_path = ckpt_overrides.get(class_name) or latest_checkpoint_for_config(config)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found for {class_name}: {ckpt_path}")
        resolved.append((class_name, config_path, ckpt_path))

    rows = []
    for class_index, (class_name, config_path, ckpt_path) in enumerate(resolved):
        generate_class(class_name, config_path, ckpt_path, args, class_index, output_root, rows)

    manifest_path = write_manifest(output_root, rows)
    print(f"[DONE] Wrote trajectory manifest: {manifest_path}")


if __name__ == "__main__":
    main()