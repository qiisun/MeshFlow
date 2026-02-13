import bpy
import sys
import os
import glob
import argparse

def clean_and_export(input_path, output_path):
    # ==============================
    # 1. 彻底清空场景
    # ==============================
    if bpy.context.active_object and bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
        
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # ==============================
    # 2. 导入 (针对 Blender 4.1+)
    # ==============================
    print(f"Importing: {input_path}")
    try:
        # Blender 4.1+ 使用这个新命令
        bpy.ops.wm.obj_import(filepath=input_path)
    except Exception as e:
        print(f"Import Failed: {e}")
        return

    # ==============================
    # 3. 稳健的物体获取策略
    # ==============================
    all_meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not all_meshes:
        print("Error: Input file contains no meshes.")
        return

    # ==============================
    # 4. 准备对象 (Join)
    # ==============================
    bpy.ops.object.select_all(action='DESELECT')
    target_obj = all_meshes[0]
    bpy.context.view_layer.objects.active = target_obj
    
    for obj in all_meshes:
        obj.select_set(True)

    if len(all_meshes) > 1:
        print("Joining meshes...")
        bpy.ops.object.join()

    target_obj = bpy.context.active_object

    # ==============================
    # 5. 核心清理逻辑 (Clean Mesh & Normals) -- 修改重点在这里
    # ==============================
    print("Processing geometry...")
    bpy.ops.object.mode_set(mode='EDIT')
    
    # --- A. 合并重叠顶点 (Remove Doubles) ---
    # 0.03 的阈值如果不合适可以调小 (例如 0.001)，以免细节丢失
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.03)

    # --- B. 删除游离的几何体 (Delete Loose) ---
    # 删除那些不构成面的孤立点或线
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()

    # --- C. 清理简并几何体 (Dissolve Degenerate) ---
    # 清理面积为0的面或长度为0的边
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)

    # --- D. 补洞 (Fill Holes) ---
    # 注意：如果模型非常碎，这步可能会产生奇怪的几何结构，视情况保留
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.edge_face_add()

    # --- E. 重算法向 (Recalculate Normals) [关键步骤] ---
    # 必须放在补洞之后，确保新补的面方向也正确
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)

    # ==============================
    # 6. 导出
    # ==============================
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 这是一个让模型看起来光滑的操作 (可选)
    # bpy.ops.object.shade_smooth() 
    
    print(f"Exporting to: {output_path}")
    try:
        # 导出设置：确保包含法向信息 (export_normals=True 默认通常是开启的)
        bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=True, export_materials=False)
    except Exception as e:
        print(f"Export Failed: {e}")

# ==============================
# 参数解析部分 (保持你的修正)
# ==============================
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = []
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, required=True, help="Folder containing .obj files")
args = parser.parse_args(argv)

input_folder = args.input_folder
output_folder = input_folder + "__cleaned"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = glob.glob(os.path.join(input_folder, "*.obj"))

print(f"Found {len(files)} OBJ files in {input_folder}")

for f in files:
    filename = os.path.basename(f)
    out_path = os.path.join(output_folder, filename)
    clean_and_export(f, out_path)