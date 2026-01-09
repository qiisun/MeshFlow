import bpy
import sys
import os
import glob

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
        # Blender 4.1 使用这个新命令
        bpy.ops.wm.obj_import(filepath=input_path)
    except Exception as e:
        print(f"Import Failed: {e}")
        return

    # ==============================
    # 3. 稳健的物体获取策略 (Fix)
    # ==============================
    # 不要依赖 context.selected_objects，因为导入器可能没选中它们。
    # 既然我们刚才清空了场景，现在场景里剩下的所有 MESH 一定是刚导入的。
    all_meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    print(f"Found {len(all_meshes)} mesh objects.")

    if not all_meshes:
        print("Error: Input file contains no meshes.")
        return

    # ==============================
    # 4. 强制构建上下文 (Context)
    # ==============================
    # 取消所有选择
    bpy.ops.object.select_all(action='DESELECT')

    # 将第一个物体设为活跃 (Active) —— 这是 mode_set 不报错的关键
    target_obj = all_meshes[0]
    bpy.context.view_layer.objects.active = target_obj
    
    # 手动选中所有网格 (为了 Join)
    for obj in all_meshes:
        obj.select_set(True)

    # 只有当大于1个物体时才执行 Join
    if len(all_meshes) > 1:
        print("Joining meshes...")
        bpy.ops.object.join()

    # Join 之后，target_obj 依然是 Active，不需要重新获取
    # 但为了保险，我们重新指向当前活跃物体
    target_obj = bpy.context.active_object

    # ==============================
    # 5. 编辑模式操作
    # ==============================
    print("Processing geometry...")
    bpy.ops.object.mode_set(mode='EDIT')
    
    # 全选点 -> Merge
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.05)
    
    # 填补破洞
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.edge_face_add()

    # ==============================
    # 6. 导出
    # ==============================
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"Exporting to: {output_path}")
    try:
        # Blender 4.1 导出命令
        bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=True)
    except Exception as e:
        print(f"Export Failed: {e}")


input_folder = "output/post/gen_50steps_cfg1"
output_folder = "output/post/blender"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = glob.glob(os.path.join(input_folder, "*.obj"))

for f in files:
    filename = os.path.basename(f)
    out_path = os.path.join(output_folder, filename)
    
    clean_and_export(f, out_path)
    
# blender -b -P post_blender.py 