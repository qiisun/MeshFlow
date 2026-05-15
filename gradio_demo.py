import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError("Gradio is required. Please run: pip install gradio") from exc

try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    pymeshlab = None
    PYMESHLAB_AVAILABLE = False

from datasets.mesh_dataset import save_mesh
from flow_matching import create_transport
from inference import (
    _configure_nccl_env_for_consumer_rtx,
    _load_val_face_lengths,
    _make_sample_fn,
    build_model_from_config,
    load_config,
)


GRADIO_TEMP_DIR = os.path.abspath(os.path.join("output", "gradio_tmp"))
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", GRADIO_TEMP_DIR)
os.environ.setdefault("TMPDIR", GRADIO_TEMP_DIR)
os.environ.setdefault("TMP", GRADIO_TEMP_DIR)
os.environ.setdefault("TEMP", GRADIO_TEMP_DIR)


DEFAULT_MAX_VAL = 1.0 / 0.3762
DEFAULT_DEVICE = "auto"
DEFAULT_NUM_SAMPLING_STEPS = 20

CATEGORY_CONFIG_MAP = {
    "bench": "configs/snet/base-120m-ot-v-bench.yaml",
    "chair": "configs/snet/base-120m-ot-v-chair.yaml",
    "lamp": "configs/snet/base-120m-ot-v-lamp.yaml",
    "table": "configs/snet/base-120m-ot-v-table.yaml",
}

CATEGORY_CKPT_MAP = {
    "bench": "output/120m-ot-v-bench/checkpoints/01000000.pt",
    "chair": "output/120m-ot-v-chair/checkpoints/01000000.pt",
    "lamp": "output/120m-ot-v-lamp/checkpoints/01000000.pt",
    "table": "output/120m-ot-v-table/checkpoints/01000000.pt",
}


@dataclass
class SamplerState:
    config_path: str
    ckpt_path: str
    device_name: str
    config: Dict
    model: torch.nn.Module
    sample_fn: callable
    max_length: int
    max_face_count_safe: int
    val_face_lengths: list


_CACHED_STATE: Optional[SamplerState] = None


def _resolve_device(device_preference: str) -> torch.device:
    if device_preference == "cpu":
        return torch.device("cpu")
    if device_preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_sampler(
    config_path: str,
    ckpt_path: str,
    device_preference: str,
    num_sampling_steps: Optional[int],
    timestep_shift: Optional[float],
) -> SamplerState:
    config = load_config(config_path)
    config["ckpt_path"] = ckpt_path
    sample_cfg = config.setdefault("sample", {})
    if num_sampling_steps is not None and num_sampling_steps > 0:
        sample_cfg["num_sampling_steps"] = int(num_sampling_steps)
    if timestep_shift is not None:
        sample_cfg["timestep_shift"] = float(timestep_shift)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _resolve_device(device_preference)

    model = build_model_from_config(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["ema"])
    model = model.to(device)
    model.eval()

    transport = create_transport(
        config["transport"]["path_type"],
        config["transport"]["prediction"],
        use_lognorm=config["transport"].get("use_lognorm", False),
        use_jit=config["transport"].get("use_jit", False),
    )

    sample_fn = _make_sample_fn(transport, config["sample"], config["sample"].get("timestep_shift", 0.0))

    max_length = int(config["model"].get("max_length", 800))
    face_bin = int(config["model"].get("face_bin", 1) or 1)
    num_classes = max_length // face_bin if face_bin > 0 else max_length
    if num_classes > 0:
        max_face_count_safe = min(max_length, num_classes * face_bin) - 1
    else:
        max_face_count_safe = max_length - 1
    max_face_count_safe = max(1, max_face_count_safe)

    val_face_lengths = _load_val_face_lengths(config)
    val_face_lengths = [x for x in val_face_lengths if 1 <= int(x) <= max_face_count_safe]

    return SamplerState(
        config_path=config_path,
        ckpt_path=ckpt_path,
        device_name=str(device),
        config=config,
        model=model,
        sample_fn=sample_fn,
        max_length=max_length,
        max_face_count_safe=max_face_count_safe,
        val_face_lengths=val_face_lengths,
    )


def _get_sampler(
    config_path: str,
    ckpt_path: str,
    device_preference: str,
    num_sampling_steps: Optional[int],
    timestep_shift: Optional[float],
) -> SamplerState:
    global _CACHED_STATE
    resolved_device = str(_resolve_device(device_preference))

    cache_hit = (
        _CACHED_STATE is not None
        and _CACHED_STATE.config_path == config_path
        and _CACHED_STATE.ckpt_path == ckpt_path
        and _CACHED_STATE.device_name == resolved_device
        and int(_CACHED_STATE.config["sample"].get("num_sampling_steps", -1)) == int(
            num_sampling_steps
            if num_sampling_steps is not None and num_sampling_steps > 0
            else _CACHED_STATE.config["sample"].get("num_sampling_steps", -1)
        )
        and float(_CACHED_STATE.config["sample"].get("timestep_shift", 0.0)) == float(
            timestep_shift
            if timestep_shift is not None
            else _CACHED_STATE.config["sample"].get("timestep_shift", 0.0)
        )
    )

    if cache_hit:
        return _CACHED_STATE

    _CACHED_STATE = _build_sampler(
        config_path=config_path,
        ckpt_path=ckpt_path,
        device_preference=device_preference,
        num_sampling_steps=num_sampling_steps,
        timestep_shift=timestep_shift,
    )
    return _CACHED_STATE


def _merge_close_vertices(mesh: trimesh.Trimesh, tolerance: float) -> Tuple[trimesh.Trimesh, int]:
    if tolerance <= 0:
        return mesh, 0

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    if vertices.shape[0] < 2:
        return mesh, 0

    tree = cKDTree(vertices)
    pairs = tree.query_pairs(r=tolerance, output_type="ndarray")
    if pairs.size == 0:
        return mesh, 0

    parent = np.arange(vertices.shape[0], dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for a, b in pairs:
        union(int(a), int(b))

    roots = np.array([find(i) for i in range(vertices.shape[0])], dtype=np.int64)
    unique_roots, inverse = np.unique(roots, return_inverse=True)

    if unique_roots.shape[0] == vertices.shape[0]:
        return mesh, 0

    new_vertices = np.zeros((unique_roots.shape[0], 3), dtype=np.float64)
    counts = np.zeros(unique_roots.shape[0], dtype=np.float64)
    np.add.at(new_vertices, inverse, vertices)
    np.add.at(counts, inverse, 1.0)
    new_vertices = (new_vertices / counts[:, None]).astype(np.float32)

    new_faces = inverse[faces]
    nondeg = (new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2]) & (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[nondeg]

    merged_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    merged_count = int(vertices.shape[0] - new_vertices.shape[0])
    return merged_mesh, merged_count


def _close_holes_with_pymeshlab(mesh: trimesh.Trimesh, max_hole_edges: int) -> trimesh.Trimesh:
    if not PYMESHLAB_AVAILABLE or max_hole_edges <= 0:
        return mesh

    try:
        ms = pymeshlab.MeshSet()
        ml_mesh = pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices), face_matrix=np.asarray(mesh.faces))
        ms.add_mesh(ml_mesh, "mesh")
        ms.meshing_close_holes(maxholesize=int(max_hole_edges))
        out = ms.current_mesh()
        return trimesh.Trimesh(vertices=out.vertex_matrix(), faces=out.face_matrix(), process=False)
    except Exception:
        return mesh


def _post_process_mesh(
    mesh: trimesh.Trimesh,
    merge_duplicates: bool,
    merge_tolerance: float,
    fill_holes: bool,
    max_hole_edges: int,
) -> Tuple[trimesh.Trimesh, Dict[str, int]]:
    stats = {
        "merged_exact": 0,
        "merged_close": 0,
        "holes_filled": 0,
    }

    before_v = int(mesh.vertices.shape[0])

    if merge_duplicates:
        mesh.merge_vertices()
        stats["merged_exact"] = max(before_v - int(mesh.vertices.shape[0]), 0)

    mesh, merged_close = _merge_close_vertices(mesh, merge_tolerance)
    stats["merged_close"] = merged_close

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if fill_holes:
        pre_watertight = bool(mesh.is_watertight)
        mesh = _close_holes_with_pymeshlab(mesh, max_hole_edges)
        trimesh.repair.fill_holes(mesh)
        post_watertight = bool(mesh.is_watertight)
        if (not pre_watertight) and post_watertight:
            stats["holes_filled"] = 1

    mesh.fix_normals()
    return mesh, stats


def generate_mesh(
    category: str,
    seed: int,
    face_count: int,
    enable_post_processing: bool,
) -> Tuple[str, str, str, str]:
    if category not in CATEGORY_CKPT_MAP:
        raise ValueError(f"Unsupported category: {category}")

    ckpt_path = CATEGORY_CKPT_MAP[category].strip()
    if not ckpt_path:
        raise ValueError(f"Checkpoint path for category '{category}' is empty")
    config_path = CATEGORY_CONFIG_MAP.get(category, CATEGORY_CONFIG_MAP["bench"])

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    _set_seed(int(seed))
    sampler = _get_sampler(config_path, ckpt_path, DEFAULT_DEVICE, DEFAULT_NUM_SAMPLING_STEPS, None)

    device = next(sampler.model.parameters()).device
    model = sampler.model
    sample_fn = sampler.sample_fn
    max_length = sampler.max_length

    safe_upper = int(sampler.max_face_count_safe)

    num_samples = 4
    if face_count <= 0:
        if sampler.val_face_lengths:
            chosen_faces = np.random.choice(sampler.val_face_lengths, size=num_samples, replace=True).astype(np.int64)
        else:
            fallback = max(32, min(safe_upper, max_length // 2))
            chosen_faces = np.full((num_samples,), fallback, dtype=np.int64)
    else:
        chosen_faces = np.full((num_samples,), int(max(1, min(face_count, safe_upper))), dtype=np.int64)

    z = torch.randn(num_samples, max_length, 9, device=device)
    z_in = torch.cat([z, z], dim=0)
    y = torch.tensor(chosen_faces, dtype=torch.long, device=device)
    mask = (torch.arange(max_length, device=device).unsqueeze(0) < y.unsqueeze(1))

    with torch.no_grad():
        cfg_scale = float(sampler.config.get("sample", {}).get("cfg_scale", 2.0))
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = sample_fn(z_in, model.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)
        else:
            out = sample_fn(z_in, model.forward_with_cfg, y=y, cfg_scale=cfg_scale, mask=mask)

    cond_samples = out[-1][:num_samples]

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join("output", "gradio_demo", f"{ts}-{time.time_ns() % 1000000:06d}")
    os.makedirs(out_dir, exist_ok=True)

    post_objs = []
    for index in range(num_samples):
        valid_tokens = cond_samples[index][: int(chosen_faces[index])].detach().cpu().numpy()
        raw_obj = os.path.join(out_dir, f"raw_{index + 1}.obj")
        post_obj = os.path.join(out_dir, f"post_{index + 1}.obj")

        save_mesh(valid_tokens, raw_obj, clean=False, max_val=DEFAULT_MAX_VAL)

        raw_mesh = trimesh.load(raw_obj, force="mesh", process=False)
        work_mesh = raw_mesh.copy()
        if enable_post_processing:
            work_mesh, _ = _post_process_mesh(
                work_mesh,
                merge_duplicates=True,
                merge_tolerance=0.0,
                fill_holes=True,
                max_hole_edges=200,
            )
        work_mesh.export(post_obj)
        post_objs.append(post_obj)

    return tuple(post_objs)


def build_demo():
    with gr.Blocks(title="MeshFlow Demo") as demo:
        gr.Markdown("# MeshFlow Demo")
        gr.Markdown(
            "Demo for SIGGRAPH 2026 Paper MeshFlow: Mesh Generation with Equivariant Flow Matching."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                category = gr.Dropdown(choices=["bench", "chair", "lamp", "table"], value="bench", label="Category")
                seed = gr.Number(value=42, precision=0, label="Seed")
                face_count = gr.Slider(minimum=0, maximum=799, value=0, step=1, label="Face Count (0 = random from val split)")
                enable_post_processing = gr.Checkbox(value=True, label="Post-processing (merge vertices + fill holes)")
                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    model_view_1 = gr.Model3D(label="Mesh 1")
                    model_view_2 = gr.Model3D(label="Mesh 2")
                with gr.Row():
                    model_view_3 = gr.Model3D(label="Mesh 3")
                    model_view_4 = gr.Model3D(label="Mesh 4")

        run_btn.click(
            fn=generate_mesh,
            inputs=[
                category,
                seed,
                face_count,
                enable_post_processing,
            ],
            outputs=[model_view_1, model_view_2, model_view_3, model_view_4],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for MeshFlow2 mesh generation")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _configure_nccl_env_for_consumer_rtx()

    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, show_error=True)
