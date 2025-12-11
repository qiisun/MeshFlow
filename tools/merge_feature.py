import os
import numpy as np
import tqdm

def merge_npy_to_npz():
    feat_dir = "./downloaded_data/dummy/feature" 
    mesh_dir = "./downloaded_data/dummy/objaverse_occ_v5_ids"
    # ===========================================

    # 获取所有 npy 文件
    if not os.path.exists(feat_dir) or not os.path.exists(mesh_dir):
        print("❌ 错误：文件夹路径不存在，请检查路径配置。")
        return

    feat_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
    print(f"🔍 找到 {len(feat_files)} 个特征文件，准备开始合并...")

    success_count = 0
    skip_count = 0

    for f_name in tqdm.tqdm(feat_files):
        try:
            parts = f_name.split('_')
            try:
                decimated_idx = parts.index('decimated')
                uid = parts[decimated_idx + 1]
            except (ValueError, IndexError):
                print(f"⚠️ 跳过：无法解析文件名的 ID: {f_name}")
                continue
            target_npz_name = f"dummy_decimated_{uid}_cow.npz"
            target_npz_path = os.path.join(mesh_dir, target_npz_name)

            if not os.path.exists(target_npz_path):
                skip_count += 1
                continue

            feat_data = np.load(os.path.join(feat_dir, f_name)) # [v, feat_dim]
            with np.load(target_npz_path, allow_pickle=True) as original_npz:
                data_dict = dict(original_npz)

            data_dict['f_feature'] = feat_data

            np.savez_compressed(target_npz_path, **data_dict)
            
            success_count += 1

        except Exception as e:
            print(f"❌ 处理文件 {f_name} 时发生错误: {e}")

    print(f"\n✅ 处理完成!")
    print(f"成功合并: {success_count} 个")
    print(f"未找到对应NPZ/跳过: {skip_count} 个")

if __name__ == "__main__":
    merge_npy_to_npz()