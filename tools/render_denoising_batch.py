#!/usr/bin/env python3
"""
Batch-render denoising trajectory folders into short videos and static PNGs.

Example:
  blender -b -P tools/render_denoising_batch.py -- \
    --input_root output/denoise10_lamp_desk_table_100/trajectories \
    --output_root output/denoise10_lamp_desk_table_100 \
    --classes lamp desk table \
    --fps 56
"""
import argparse
import csv
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bpy  # noqa: E402

from render_video import (  # noqa: E402
    DEFAULT_BODY_COLOR,
    DEFAULT_WIRE_THICKNESS,
    blender_argv,
    clear_render_meshes,
    collect_mesh_paths,
    compute_camera_dimension,
    create_black_wire_material,
    encode_video,
    parse_rgb,
    render_still,
    setup_camera_and_light,
    setup_scene,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render 10-step denoising trajectories in batch.")
    parser.add_argument("--input_root", required=True, help="Root containing <class>/<sample_id> trajectory dirs")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--pattern", default="*.obj")
    parser.add_argument("--expected_frames", type=int, default=10)
    parser.add_argument("--fps", type=int, default=56, help="10 frames at 56 fps gives about 0.18s per object")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--camera_angle", type=float, default=-30.0)
    parser.add_argument("--camera_tilt", type=float, default=30.0)
    parser.add_argument("--camera_margin", type=float, default=1.8)
    camera_projection = parser.add_mutually_exclusive_group()
    camera_projection.add_argument("--orthographic", dest="orthographic", action="store_true")
    camera_projection.add_argument("--perspective", dest="orthographic", action="store_false")
    parser.set_defaults(orthographic=False)
    parser.add_argument("--no_axis_fix", action="store_true")
    parser.add_argument("--body_color", type=parse_rgb, default=parse_rgb(DEFAULT_BODY_COLOR))
    parser.add_argument("--wire_color", type=parse_rgb, default=parse_rgb("0.0,0.0,0.0"))
    parser.add_argument("--background_color", type=parse_rgb, default=parse_rgb("1.0,1.0,1.0"))
    parser.add_argument("--wire_thickness", type=float, default=DEFAULT_WIRE_THICKNESS)
    parser.add_argument("--transparent_frames", action="store_true")
    parser.add_argument("--cleanup_frames", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_objects", type=int, default=0, help="Debug limit per class; 0 means all")
    parser.add_argument("--verbose_blender", action="store_true", help="Show Blender per-frame render logs")
    return parser.parse_args(blender_argv())


def discover_classes(input_root):
    return sorted(path.name for path in Path(input_root).iterdir() if path.is_dir())


def sample_dirs(input_root, class_name, max_objects):
    class_root = Path(input_root) / class_name
    if not class_root.is_dir():
        raise FileNotFoundError(f"missing trajectory class directory: {class_root}")
    dirs = sorted([path for path in class_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    if max_objects > 0:
        return dirs[:max_objects]
    return dirs


def ensure_dirs(output_root, class_name):
    paths = {
        "videos": Path(output_root) / "videos" / class_name,
        "static": Path(output_root) / "static" / class_name,
        "frames": Path(output_root) / "frames" / class_name,
        "manifests": Path(output_root) / "manifests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def encode_frames_with_ffmpeg(frame_dir, output_path, fps):
    candidates = [os.environ.get("FFMPEG_BINARY"), "/usr/bin/ffmpeg", shutil.which("ffmpeg")]
    for ffmpeg in [candidate for candidate in candidates if candidate]:
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-start_number",
            "1",
            "-i",
            os.path.join(frame_dir, "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            return True
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"[WARN] ffmpeg encode failed with {ffmpeg}: {exc}")
    return False


def configure_cycles_device():
    try:
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.render.use_persistent_data = True
        preferences = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA"):
            try:
                preferences.compute_device_type = device_type
                preferences.get_devices()
                enabled = False
                for device in preferences.devices:
                    device.use = device.type != "CPU"
                    enabled = enabled or device.use
                if enabled:
                    return device_type
            except Exception:
                continue
    except Exception:
        pass
    return "CPU"


@contextlib.contextmanager
def suppress_blender_output(enabled):
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


def render_sample(sample_dir, video_path, static_path, frame_dir, args):
    mesh_paths = collect_mesh_paths(str(sample_dir), args.pattern)
    if len(mesh_paths) != args.expected_frames:
        raise RuntimeError(f"{sample_dir} has {len(mesh_paths)} frames, expected {args.expected_frames}")

    frame_dir.mkdir(parents=True, exist_ok=True)
    setup_scene(args)
    configure_cycles_device()
    material = create_black_wire_material(args.body_color, args.wire_color, args.wire_thickness)

    with suppress_blender_output(not args.verbose_blender):
        final_dimension = compute_camera_dimension([mesh_paths[-1]], material, apply_axis_fix=not args.no_axis_fix)
        setup_camera_and_light(final_dimension, args)

        rendered_frames = []
        for index, mesh_path in enumerate(mesh_paths, start=1):
            frame_path = frame_dir / f"frame_{index:06d}.png"
            render_still(
                mesh_path,
                str(frame_path),
                material,
                apply_axis_fix=not args.no_axis_fix,
                target_dimension=final_dimension,
            )
            rendered_frames.append(str(frame_path))

    shutil.copyfile(rendered_frames[-1], static_path)
    if not encode_frames_with_ffmpeg(str(frame_dir), video_path, args.fps):
        encode_video(rendered_frames, str(video_path), args.fps, args.resolution, args.background_color)
    if args.cleanup_frames:
        shutil.rmtree(frame_dir)
    clear_render_meshes()


def write_manifest(rows, output_root, class_name):
    manifest_path = Path(output_root) / "manifests" / f"render_{class_name}.csv"
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["class", "sample_id", "trajectory_dir", "video", "static", "frames_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main():
    args = parse_args()
    classes = args.classes or discover_classes(args.input_root)
    total = 0

    for class_name in classes:
        paths = ensure_dirs(args.output_root, class_name)
        dirs = sample_dirs(args.input_root, class_name, args.max_objects)
        print(f"[INFO] {class_name}: rendering {len(dirs)} samples")
        rows = []
        for index, sample_dir in enumerate(dirs, start=1):
            sample_id = sample_dir.name
            video_path = paths["videos"] / f"{sample_id}.mp4"
            static_path = paths["static"] / f"{sample_id}.png"
            frame_dir = paths["frames"] / sample_id
            if args.skip_existing and video_path.exists() and static_path.exists():
                print(f"[SKIP] {class_name}/{sample_id}")
            else:
                print(f"[INFO] {class_name}: {index}/{len(dirs)} sample={sample_id}")
                render_sample(sample_dir, video_path, static_path, frame_dir, args)
            rows.append(
                {
                    "class": class_name,
                    "sample_id": sample_id,
                    "trajectory_dir": str(sample_dir),
                    "video": str(video_path),
                    "static": str(static_path),
                    "frames_dir": str(frame_dir),
                }
            )
            total += 1
        manifest_path = write_manifest(rows, args.output_root, class_name)
        print(f"[DONE] {class_name}: manifest={manifest_path}")

    print(f"[DONE] Rendered {total} samples")


if __name__ == "__main__":
    main()