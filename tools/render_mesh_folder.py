#!/usr/bin/env python3
"""
Render OBJ/PLY meshes from a folder into PNG thumbnails and an HTML gallery.

Example:
  blender -b -P tools/render_mesh_folder.py -- \
    --input output/render-pipeline-denoise/infer_01000000_denoise \
    --output_dir output/render-pipeline-denoise/picks \
    --pattern "*.obj,*.ply"
"""
import argparse
import csv
import glob
import html
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_video import (  # noqa: E402
    DEFAULT_BODY_COLOR,
    DEFAULT_WIRE_THICKNESS,
    blender_argv,
    clear_render_meshes,
    compute_camera_dimension,
    create_black_wire_material,
    import_mesh,
    natural_key,
    object_max_dimension,
    parse_rgb,
    prepare_mesh_object,
    setup_camera_and_light,
    setup_scene,
)

import bpy  # noqa: E402


SUPPORTED_PICK_EXTENSIONS = {".obj", ".ply"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Mesh folder, glob, or single .obj/.ply file")
    parser.add_argument("--output_dir", required=True, help="Directory for PNG renders, manifest.csv, and index.html")
    parser.add_argument(
        "--pattern",
        "--patterns",
        action="append",
        default=None,
        help="Glob pattern for folder input. Repeat or comma-separate. Default: *.obj,*.ply",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input folder recursively")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth mesh after natural sorting")
    parser.add_argument("--max_meshes", "--limit", dest="max_meshes", type=int, default=0, help="Render at most N meshes; 0 means all")
    parser.add_argument("--skip_existing", action="store_true", help="Do not rerender PNG files that already exist")
    parser.add_argument("--no_gallery", action="store_true", help="Skip writing index.html")

    parser.add_argument("--resolution", type=int, default=512, help="Square render resolution")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per image")
    parser.add_argument("--fps", type=int, default=24, help=argparse.SUPPRESS)
    parser.add_argument("--camera_fit", choices=["each", "all"], default="each", help="Fit each mesh or use one shared camera scale")
    parser.add_argument("--camera_angle", type=float, default=-30.0, help="Camera azimuth in degrees")
    parser.add_argument("--camera_tilt", type=float, default=30.0, help="Camera elevation in degrees")
    parser.add_argument("--camera_margin", type=float, default=2.4)
    camera_projection = parser.add_mutually_exclusive_group()
    camera_projection.add_argument("--orthographic", dest="orthographic", action="store_true", help="Use an orthographic camera")
    camera_projection.add_argument("--perspective", dest="orthographic", action="store_false", help="Use a perspective camera")
    parser.set_defaults(orthographic=True)

    mesh_scale = parser.add_mutually_exclusive_group()
    mesh_scale.add_argument("--normalize_each_mesh", dest="normalize_each_mesh", action="store_true", help="Scale each mesh to fill its thumbnail")
    mesh_scale.add_argument("--preserve_scale", dest="normalize_each_mesh", action="store_false", help="Preserve original mesh scale")
    parser.set_defaults(normalize_each_mesh=True)

    parser.add_argument("--no_axis_fix", action="store_true", help="Disable x=90 degree rotation used by the MeshFlow render style")
    parser.add_argument("--body_color", type=parse_rgb, default=parse_rgb(DEFAULT_BODY_COLOR))
    parser.add_argument("--wire_color", type=parse_rgb, default=parse_rgb("0.0,0.0,0.0"))
    parser.add_argument("--background_color", type=parse_rgb, default=parse_rgb("1.0,1.0,1.0"))
    parser.add_argument("--wire_thickness", type=float, default=DEFAULT_WIRE_THICKNESS)
    parser.add_argument("--transparent_frames", action="store_true", help="Render PNGs with alpha")
    return parser.parse_args(blender_argv())


def split_patterns(pattern_groups):
    if pattern_groups is None:
        return ["*.obj", "*.ply"]

    patterns = []
    for group in pattern_groups:
        patterns.extend(pattern.strip() for pattern in group.split(",") if pattern.strip())
    return patterns or ["*.obj", "*.ply"]


def collect_mesh_paths(input_path, patterns, recursive):
    input_path = os.path.abspath(input_path)
    if os.path.isdir(input_path):
        paths = []
        for pattern in patterns:
            search_pattern = os.path.join(input_path, "**", pattern) if recursive else os.path.join(input_path, pattern)
            paths.extend(glob.glob(search_pattern, recursive=recursive))
    elif glob.has_magic(input_path):
        paths = glob.glob(input_path, recursive=recursive)
    else:
        paths = [input_path]

    filtered_paths = []
    seen_paths = set()
    for path in paths:
        resolved_path = os.path.abspath(path)
        if Path(resolved_path).suffix.lower() not in SUPPORTED_PICK_EXTENSIONS:
            continue
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        filtered_paths.append(resolved_path)

    filtered_paths = sorted(filtered_paths, key=natural_key)
    if not filtered_paths:
        raise FileNotFoundError(f"no .obj/.ply files found for input: {input_path}")
    return filtered_paths


def select_paths(paths, stride, max_meshes):
    selected_paths = paths[:: max(1, stride)]
    if max_meshes > 0:
        selected_paths = selected_paths[:max_meshes]
    return selected_paths


def clear_cameras_and_lights():
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def input_root_for_names(input_path):
    input_path = os.path.abspath(input_path)
    return Path(input_path).resolve() if os.path.isdir(input_path) else None


def safe_output_stem(mesh_path, input_root, index):
    path = Path(mesh_path).resolve()
    if input_root is not None:
        try:
            label = path.relative_to(input_root).with_suffix("").as_posix().replace("/", "__")
        except ValueError:
            label = path.stem
    else:
        label = path.stem
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or path.stem
    return f"{index:04d}_{safe_label}"


def output_image_path(mesh_path, input_root, output_dir, index):
    return os.path.join(output_dir, safe_output_stem(mesh_path, input_root, index) + ".png")


def target_dimension_for_render(args, shared_dimension):
    if not args.normalize_each_mesh:
        return None
    if args.camera_fit == "all":
        return shared_dimension
    return 1.0


def render_mesh(mesh_path, output_path, material, args, shared_dimension=None):
    clear_render_meshes()
    obj = prepare_mesh_object(
        import_mesh(mesh_path),
        material,
        apply_axis_fix=not args.no_axis_fix,
        target_dimension=target_dimension_for_render(args, shared_dimension),
    )
    fit_dimension = shared_dimension if args.camera_fit == "all" else object_max_dimension(obj)
    clear_cameras_and_lights()
    setup_camera_and_light(max(fit_dimension, 1.0), args)
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def write_manifest(rows, output_dir):
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["index", "source", "image"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def write_gallery(rows, output_dir):
    gallery_path = os.path.join(output_dir, "index.html")
    items = []
    for row in rows:
        image_name = html.escape(os.path.basename(row["image"]))
        source_name = html.escape(row["source"])
        items.append(
            "<figure>"
            f'<a href="{image_name}"><img src="{image_name}" loading="lazy" alt="{source_name}"></a>'
            f"<figcaption>{row['index']:04d} {source_name}</figcaption>"
            "</figure>"
        )

    with open(gallery_path, "w") as gallery_file:
        gallery_file.write(
            "<!doctype html>\n"
            "<html><head><meta charset=\"utf-8\">\n"
            "<title>Mesh render picks</title>\n"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:24px;background:#fff;color:#111}"
            "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px}"
            "figure{margin:0;border:1px solid #ddd;padding:8px}"
            "img{width:100%;height:auto;display:block;background:#fff}"
            "figcaption{font-size:12px;line-height:1.35;margin-top:8px;word-break:break-all}"
            "</style></head><body>\n"
            f"<h1>Mesh render picks ({len(rows)})</h1>\n"
            f"<main>\n{''.join(items)}\n</main>\n"
            "</body></html>\n"
        )
    return gallery_path


def main():
    args = parse_args()
    mesh_paths = select_paths(collect_mesh_paths(args.input, split_patterns(args.pattern), args.recursive), args.stride, args.max_meshes)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Rendering {len(mesh_paths)} mesh thumbnails")
    print(f"[INFO] Output directory: {output_dir}")

    setup_scene(args)
    material = create_black_wire_material(args.body_color, args.wire_color, args.wire_thickness)
    shared_dimension = None
    if args.camera_fit == "all":
        shared_dimension = compute_camera_dimension(mesh_paths, material, apply_axis_fix=not args.no_axis_fix)

    input_root = input_root_for_names(args.input)
    rows = []
    for index, mesh_path in enumerate(mesh_paths):
        image_path = output_image_path(mesh_path, input_root, output_dir, index)
        if args.skip_existing and os.path.exists(image_path):
            print(f"[SKIP] {mesh_path}")
        else:
            print(f"[INFO] Rendering {index + 1}/{len(mesh_paths)}: {mesh_path}")
            render_mesh(mesh_path, image_path, material, args, shared_dimension=shared_dimension)
        rows.append({"index": index, "source": mesh_path, "image": image_path})

    clear_render_meshes()
    manifest_path = write_manifest(rows, output_dir)
    print(f"[DONE] Wrote manifest: {manifest_path}")
    if not args.no_gallery:
        gallery_path = write_gallery(rows, output_dir)
        print(f"[DONE] Wrote gallery: {gallery_path}")


if __name__ == "__main__":
    main()