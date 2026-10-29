import os
import shutil
from pathlib import Path

root = Path(os.environ.get('NIF_DIOR_SOURCE', 'data/DIOR-R'))
source_folder = root / 'Annotations/Oriented Bounding Boxes TXT'
target_root = Path(os.environ.get('NIF_DATA_ROOT', 'data')) / 'dior'
trainval_target_folder = target_root / 'trainval/annfiles'
test_target_folder = target_root / 'test/annfiles'
os.makedirs(trainval_target_folder, exist_ok=True)
os.makedirs(test_target_folder, exist_ok=True)

# 读取train.txt文件中的文件名列表
with open(root / 'ImageSets/Main/train.txt', 'r') as file:
    file_names = file.read().splitlines()

# 遍历文件名列表并复制文件
total_trainval = 0
for file_name in file_names:
    source_file_path = os.path.join(source_folder, f"{file_name}.txt")
    target_file_path = os.path.join(trainval_target_folder, f"{file_name}.txt")

    # 使用shutil库进行文件复制
    if os.path.exists(source_file_path):
        total_trainval += 1
        shutil.copy(source_file_path, target_file_path)
        print(f"复制 {file_name}.txt 完成")
print("train 复制完成！")

# 读取val.txt文件中的文件名列表
with open(root / 'ImageSets/Main/val.txt', 'r') as file:
    file_names = file.read().splitlines()

for file_name in file_names:
    source_file_path = os.path.join(source_folder, f"{file_name}.txt")
    target_file_path = os.path.join(trainval_target_folder, f"{file_name}.txt")

    # 使用shutil库进行文件复制
    if os.path.exists(source_file_path):
        total_trainval += 1
        shutil.copy(source_file_path, target_file_path)
        print(f"复制 {file_name}.txt 完成")
print("val 复制完成！")
print(f"total trainval {total_trainval}")

# 读取test.txt文件中的文件名列表
with open(root / 'ImageSets/Main/test.txt', 'r') as file:
    file_names = file.read().splitlines()

# 遍历文件名列表并复制文件
total_test = 0
for file_name in file_names:
    source_file_path = os.path.join(source_folder, f"{file_name}.txt")
    target_file_path = os.path.join(test_target_folder, f"{file_name}.txt")

    # 使用shutil库进行文件复制
    if os.path.exists(source_file_path):
        total_test += 1
        shutil.copy(source_file_path, target_file_path)
        print(f"复制 {file_name}.txt 完成")

print("test 复制完成！")
print(f"total test {total_test}")

print(f"total trainval {total_trainval}")
print(f"total test {total_test}")
