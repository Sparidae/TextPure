import os


def find_files_with_unique_sizes(folder_path):
    """
    找出指定文件夹内大小独特的文件。

    Args:
        folder_path (str): 要检查的文件夹路径。

    Returns:
        list: 包含大小独特的文件完整路径的列表。
    """
    file_sizes = {}  # 字典，key为文件大小，value为该大小对应的文件路径列表

    print(f"正在扫描文件夹：{folder_path}")

    try:
        # 遍历文件夹中的所有条目
        for entry_name in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry_name)

            # 只处理文件，排除子文件夹
            if os.path.isfile(entry_path):
                try:
                    file_size = os.path.getsize(entry_path)

                    # 将文件路径添加到对应大小的列表中
                    if file_size not in file_sizes:
                        file_sizes[file_size] = []
                    file_sizes[file_size].append(entry_path)

                except OSError as e:
                    print(f"无法获取文件大小 {entry_path}: {e}")
                    continue

    except FileNotFoundError:
        print(f"错误：文件夹未找到：{folder_path}")
        return []
    except PermissionError:
        print(f"错误：没有权限访问文件夹：{folder_path}")
        return []

    unique_size_files = []
    # 遍历文件大小分组，找出只有一个文件的分组
    for size, files_list in file_sizes.items():
        if len(files_list) == 1:
            unique_size_files.extend(files_list)  # 将这个唯一的文件添加到结果列表

    return unique_size_files


# --- 使用示例 ---
folder_to_check = "results/crawl_20250509_113941"  # <-- 将这里替换为你的实际文件夹路径

unique_files = find_files_with_unique_sizes(folder_to_check)

if unique_files:
    print("\n以下文件的大小在文件夹中是独特的：")
    for file_path in unique_files:
        print(file_path)
else:
    print("\n没有找到大小独特的文件（所有文件大小都有重复）。")
