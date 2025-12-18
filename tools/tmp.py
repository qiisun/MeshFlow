import os
import numpy as np
import zipfile
from tqdm import tqdm

# !!! 修改这里为你的数据文件夹路径 !!!
dataset_root = "./downloaded_data/dummy" 

print(f"开始扫描并清理坏文件: {dataset_root}")

bad_files = []
all_files = []

# 1. 遍历所有文件
for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if file.endswith(".npz"):
            all_files.append(os.path.join(root, file))

print(f"总共找到 {len(all_files)} 个 .npz 文件")

# 2. 逐个检查
for file_path in tqdm(all_files, desc="Checking"):
    try:
        # 尝试加载，只读 header 不读全部数据，速度快
        with np.load(file_path) as data:
            # 尝试访问一个 key 确认文件没烂
            _ = data.files 
    except (zipfile.BadZipFile, ValueError, OSError, EOFError) as e:
        # 如果报错，就是坏文件
        bad_files.append(file_path)
        # print(f"发现坏文件: {file_path} | 错误: {e}")

# 3. 汇报结果
print(f"\n扫描结束！")
print(f"正常文件: {len(all_files) - len(bad_files)}")
print(f"损坏文件: {len(bad_files)}")

# 4. 执行删除
if len(bad_files) > 0:
    print("\n准备删除以下坏文件（前5个示例）:")
    for f in bad_files[:5]:
        print(f" - {f}")
    
    choice = input(f"\n确认删除这 {len(bad_files)} 个损坏文件吗? (输入 'yes' 确认): ")
    
    if choice.strip().lower() == 'yes':
        for f in tqdm(bad_files, desc="Deleting"):
            try:
                os.remove(f)
            except OSError as e:
                print(f"删除失败: {f} : {e}")
        print("清理完成！现在剩下的都是好文件了。")
    else:
        print("已取消删除。")
else:
    print("恭喜，没有发现坏文件！")