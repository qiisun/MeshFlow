#!/usr/bin/env python3
"""
Render a denoising mesh sequence followed by a rotating final mesh.

Example:
  blender -b -P tools/render_video.py -- \
    --input output/120m-ot-v-bench/infer_00500000 \
    --pattern "*.obj" \
    --output output/render_mflow.mp4

The input can be a directory, a glob, or a single mesh file. If --final_mesh is
not provided, the last mesh in the sorted input sequence is used for rotation.
"""
import argparse
import glob
import math
import os
import re
import shutil
import sys
from pathlib import Path

import bpy
import mathutils


SUPPORTED_MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".glb", ".gltf"}
DEFAULT_BODY_COLOR = "0.9,0.9,0.9"
DEFAULT_WIRE_THICKNESS = 0.018
BODY_EMISSION_STRENGTH = 0.25
WORLD_BACKGROUND_STRENGTH = 1.25
CAMERA_LIGHT_ENERGY = 1000.0


def blender_argv():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def natural_key(path):
    name = os.path.basename(path)
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", name)]


def parse_rgb(value):
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B")
    if any(part < 0.0 or part > 1.0 for part in parts):
        raise argparse.ArgumentTypeError("colors must be in [0, 1]")
    return tuple(parts) + (1.0,)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Mesh sequence directory, glob, or single mesh file")
    parser.add_argument("--pattern", default="*.obj", help="Glob pattern when --input is a directory")
    parser.add_argument("--output", default="render_mflow.mp4", help="Output .mp4 or .mov path")
    parser.add_argument("--final_mesh", default=None, help="Mesh to use for final rotation; defaults to last sequence frame")
    parser.add_argument("--frames_dir", default=None, help="Directory for rendered PNG frames")
    parser.add_argument("--cleanup_frames", action="store_true", help="Delete PNG frames after video encoding")

    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=800, help="Square render resolution")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples per frame")
    parser.add_argument("--denoise_stride", type=int, default=1, help="Use every Nth denoising mesh")
    parser.add_argument("--max_denoise_frames", type=int, default=0, help="Evenly subsample denoising frames; 0 means all")
    parser.add_argument("--denoise_hold_frames", type=int, default=1, help="Repeat each denoising frame N times")
    parser.add_argument("--hold_final_frames", type=int, default=12, help="Still frames before final rotation")
    parser.add_argument("--rotate_frames", type=int, default=96, help="Number of rotation frames")
    parser.add_argument("--rotate_degrees", type=float, default=360.0)

    parser.add_argument("--camera_fit", choices=["sequence", "final"], default="final")
    parser.add_argument("--camera_angle", type=float, default=-30.0, help="Camera azimuth in degrees")
    parser.add_argument("--camera_tilt", type=float, default=30.0, help="Camera elevation in degrees")
    parser.add_argument("--camera_margin", type=float, default=1.8)
    camera_projection = parser.add_mutually_exclusive_group()
    camera_projection.add_argument("--orthographic", dest="orthographic", action="store_true", help="Use an orthographic camera")
    camera_projection.add_argument("--perspective", dest="orthographic", action="store_false", help="Use a perspective camera")
    parser.set_defaults(orthographic=False)

    frame_scale = parser.add_mutually_exclusive_group()
    frame_scale.add_argument("--normalize_each_frame", dest="normalize_each_frame", action="store_true", help="Scale every denoising frame to final mesh size")
    frame_scale.add_argument("--preserve_scale", dest="normalize_each_frame", action="store_false", help="Preserve each frame's original scale")
    parser.set_defaults(normalize_each_frame=True)

    parser.add_argument("--no_axis_fix", action="store_true", help="Disable x=90 degree rotation used by the MeshFlow render style")
    parser.add_argument("--body_color", type=parse_rgb, default=parse_rgb(DEFAULT_BODY_COLOR))
    parser.add_argument("--wire_color", type=parse_rgb, default=parse_rgb("0.0,0.0,0.0"))
    parser.add_argument("--background_color", type=parse_rgb, default=parse_rgb("1.0,1.0,1.0"))
    parser.add_argument("--wire_thickness", type=float, default=DEFAULT_WIRE_THICKNESS)
    parser.add_argument("--transparent_frames", action="store_true", help="Render intermediate PNG frames with alpha")
    return parser.parse_args(blender_argv())


def collect_mesh_paths(input_path, pattern):
    if os.path.isdir(input_path):
        paths = glob.glob(os.path.join(input_path, pattern))
    elif glob.has_magic(input_path):
        paths = glob.glob(input_path)
    else:
        paths = [input_path]

    paths = [os.path.abspath(path) for path in paths if Path(path).suffix.lower() in SUPPORTED_MESH_EXTENSIONS]
    paths = sorted(paths, key=natural_key)
    if not paths:
        raise FileNotFoundError(f"no supported mesh files found for input: {input_path}")
    return paths


def subsample_paths(paths, stride, max_count):
    stride = max(1, stride)
    paths = paths[::stride]
    if max_count <= 0 or len(paths) <= max_count:
        return paths
    if max_count == 1:
        return [paths[-1]]
    last_index = len(paths) - 1
    indices = [round(index * last_index / (max_count - 1)) for index in range(max_count)]
    return [paths[index] for index in indices]


def set_node_input(node, names, value):
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def create_black_wire_material(body_color, wire_color, wire_thickness):
    mat_name = "GreyBaseBlackWire"
    mat = bpy.data.materials.get(mat_name)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)
    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (400, 0)
    body_add = nodes.new("ShaderNodeAddShader")
    body_add.location = (200, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 100)
    bsdf.inputs["Base Color"].default_value = body_color
    bsdf.inputs["Roughness"].default_value = 0.5
    set_node_input(bsdf, ["Specular IOR Level", "Specular"], 0.2)

    body_emission = nodes.new("ShaderNodeEmission")
    body_emission.location = (0, -80)
    body_emission.inputs["Color"].default_value = body_color
    body_emission.inputs["Strength"].default_value = BODY_EMISSION_STRENGTH

    wire_emission = nodes.new("ShaderNodeEmission")
    wire_emission.location = (200, -220)
    wire_emission.inputs["Color"].default_value = wire_color
    wire_emission.inputs["Strength"].default_value = 1.0

    wire = nodes.new("ShaderNodeWireframe")
    wire.location = (-200, 100)
    wire.inputs["Size"].default_value = wire_thickness
    wire.use_pixel_size = False

    links.new(bsdf.outputs["BSDF"], body_add.inputs[0])
    links.new(body_emission.outputs["Emission"], body_add.inputs[1])
    links.new(wire.outputs["Fac"], mix.inputs["Fac"])
    links.new(body_add.outputs["Shader"], mix.inputs[1])
    links.new(wire_emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return mat


def setup_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_color_management(scene)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.fps = args.fps
    scene.render.film_transparent = args.transparent_frames
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if args.transparent_frames else "RGB"

    scene.use_nodes = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = args.background_color
    background.inputs["Strength"].default_value = WORLD_BACKGROUND_STRENGTH


def configure_color_management(scene):
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def clear_render_meshes():
    for obj in list(bpy.context.scene.objects):
        if obj.type in {"MESH", "EMPTY"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def import_mesh(path):
    before = set(bpy.data.objects)
    suffix = Path(path).suffix.lower()

    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            try:
                bpy.ops.preferences.addon_enable(module="io_scene_obj")
            except Exception:
                pass
            bpy.ops.import_scene.obj(filepath=path)
    elif suffix == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError(f"unsupported mesh format: {path}")

    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"no mesh object imported from {path}")

    if len(imported) == 1:
        return imported[0]

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    bpy.ops.object.join()
    return bpy.context.object


def object_max_dimension(obj):
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
    max_corner = mathutils.Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
    size = max_corner - min_corner
    return max(size.x, size.y, size.z)


def prepare_mesh_object(obj, material, apply_axis_fix=True, target_dimension=None):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)
    if apply_axis_fix:
        obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.context.view_layer.update()

    if target_dimension is not None:
        current_dimension = object_max_dimension(obj)
        if current_dimension > 0:
            scale = target_dimension / current_dimension
            obj.scale = (obj.scale.x * scale, obj.scale.y * scale, obj.scale.z * scale)

    obj.data.materials.clear()
    obj.data.materials.append(material)
    bpy.ops.object.shade_flat()
    bpy.context.view_layer.update()
    return obj


def compute_camera_dimension(paths, material, apply_axis_fix):
    max_dimension = 0.0
    for path in paths:
        clear_render_meshes()
        obj = prepare_mesh_object(import_mesh(path), material, apply_axis_fix=apply_axis_fix)
        max_dimension = max(max_dimension, object_max_dimension(obj))
    clear_render_meshes()
    return max(max_dimension, 1.0)


def setup_camera_and_light(max_dimension, args):
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

    azimuth = math.radians(args.camera_angle)
    elevation = math.radians(args.camera_tilt)
    distance = max_dimension * args.camera_margin
    offset = mathutils.Vector((
        -math.sin(azimuth) * math.cos(elevation),
        -math.cos(azimuth) * math.cos(elevation),
        math.sin(elevation),
    )).normalized() * distance
    camera.location = offset
    camera.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_end = max(1000.0, distance * 10.0)

    if args.orthographic:
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = max_dimension * args.camera_margin

    light_data = bpy.data.lights.new(name="CameraLight", type="POINT")
    light = bpy.data.objects.new(name="CameraLight", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = camera.location
    light_data.energy = CAMERA_LIGHT_ENERGY
    light_data.shadow_soft_size = 5.0


def render_still(mesh_path, output_path, material, apply_axis_fix, target_dimension=None):
    clear_render_meshes()
    prepare_mesh_object(import_mesh(mesh_path), material, apply_axis_fix=apply_axis_fix, target_dimension=target_dimension)
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def render_rotation(mesh_path, frame_paths, material, apply_axis_fix, rotate_degrees, target_dimension=None):
    clear_render_meshes()
    obj = prepare_mesh_object(import_mesh(mesh_path), material, apply_axis_fix=apply_axis_fix, target_dimension=target_dimension)
    rotator = bpy.data.objects.new("FinalMeshRotator", None)
    bpy.context.collection.objects.link(rotator)
    obj.parent = rotator

    total_frames = len(frame_paths)
    for frame_offset, output_path in enumerate(frame_paths):
        angle = math.radians(rotate_degrees) * frame_offset / max(1, total_frames)
        rotator.rotation_euler = (0.0, 0.0, angle)
        bpy.context.view_layer.update()
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

    clear_render_meshes()


def configure_video_encoding(scene, output_path, fps):
    scene.render.fps = fps
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = fps

    suffix = Path(output_path).suffix.lower()
    scene.render.ffmpeg.format = "QUICKTIME" if suffix == ".mov" else "MPEG4"


def encode_video(frame_paths, output_path, fps, resolution, background_color):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    configure_color_management(scene)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.frame_start = 1
    scene.frame_end = len(frame_paths)
    scene.sequence_editor_create()

    background = scene.sequence_editor.sequences.new_effect(
        name="background",
        type="COLOR",
        channel=1,
        frame_start=1,
        frame_end=len(frame_paths) + 1,
    )
    background.color = background_color[:3]

    first_frame = Path(frame_paths[0])
    frame_dir = first_frame.parent
    strip = scene.sequence_editor.sequences.new_image(
        name="rendered_frames",
        filepath=str(first_frame),
        channel=2,
        frame_start=1,
    )
    for frame_path in frame_paths[1:]:
        frame_file = Path(frame_path)
        if frame_file.parent != frame_dir:
            raise ValueError("all video frames must live in the same directory")
        strip.elements.append(frame_file.name)
    strip.frame_final_duration = len(frame_paths)
    if hasattr(strip, "blend_type"):
        strip.blend_type = "ALPHA_OVER"

    if hasattr(scene.render, "use_sequencer"):
        scene.render.use_sequencer = True
    configure_video_encoding(scene, output_path, fps)
    bpy.ops.render.render(animation=True)


def frame_path(frames_dir, frame_number):
    return os.path.join(frames_dir, f"frame_{frame_number:06d}.png")


def main():
    args = parse_args()
    sequence_paths = subsample_paths(collect_mesh_paths(args.input, args.pattern), args.denoise_stride, args.max_denoise_frames)
    final_mesh = os.path.abspath(args.final_mesh) if args.final_mesh else sequence_paths[-1]
    if not os.path.exists(final_mesh):
        raise FileNotFoundError(f"final mesh does not exist: {final_mesh}")

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_dir = args.frames_dir or os.path.splitext(output_path)[0] + "_frames"
    frames_dir = os.path.abspath(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"[INFO] Denoising meshes: {len(sequence_paths)}")
    print(f"[INFO] Final mesh: {final_mesh}")
    print(f"[INFO] Frame directory: {frames_dir}")

    setup_scene(args)
    material = create_black_wire_material(args.body_color, args.wire_color, args.wire_thickness)
    final_dimension = compute_camera_dimension([final_mesh], material, apply_axis_fix=not args.no_axis_fix)
    if args.normalize_each_frame:
        target_dimension = final_dimension
        max_dimension = final_dimension
    else:
        target_dimension = None
        camera_fit_paths = sequence_paths if args.camera_fit == "sequence" else [final_mesh]
        max_dimension = compute_camera_dimension(camera_fit_paths, material, apply_axis_fix=not args.no_axis_fix)
    setup_camera_and_light(max_dimension, args)

    rendered_frames = []
    next_frame = 1
    denoise_hold_frames = max(1, args.denoise_hold_frames)

    for mesh_index, mesh_path in enumerate(sequence_paths):
        output_frame = frame_path(frames_dir, next_frame)
        print(f"[INFO] Rendering denoise mesh {mesh_index + 1}/{len(sequence_paths)}: {os.path.basename(mesh_path)}")
        render_still(mesh_path, output_frame, material, apply_axis_fix=not args.no_axis_fix, target_dimension=target_dimension)
        rendered_frames.append(output_frame)
        next_frame += 1

        for _ in range(1, denoise_hold_frames):
            duplicate_frame = frame_path(frames_dir, next_frame)
            shutil.copyfile(output_frame, duplicate_frame)
            rendered_frames.append(duplicate_frame)
            next_frame += 1

    for hold_index in range(args.hold_final_frames):
        output_frame = frame_path(frames_dir, next_frame)
        print(f"[INFO] Rendering final hold frame {hold_index + 1}/{args.hold_final_frames}")
        render_still(final_mesh, output_frame, material, apply_axis_fix=not args.no_axis_fix, target_dimension=target_dimension)
        rendered_frames.append(output_frame)
        next_frame += 1

    rotation_frame_paths = []
    for rotate_index in range(args.rotate_frames):
        rotation_frame_paths.append(frame_path(frames_dir, next_frame + rotate_index))
    print(f"[INFO] Rendering {len(rotation_frame_paths)} rotation frames")
    render_rotation(final_mesh, rotation_frame_paths, material, not args.no_axis_fix, args.rotate_degrees, target_dimension=target_dimension)
    rendered_frames.extend(rotation_frame_paths)

    print(f"[INFO] Encoding video: {output_path}")
    encode_video(rendered_frames, output_path, args.fps, args.resolution, args.background_color)

    if args.cleanup_frames:
        shutil.rmtree(frames_dir)

    print(f"[DONE] Saved video to {output_path}")


if __name__ == "__main__":
    main()