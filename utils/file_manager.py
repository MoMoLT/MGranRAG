import os, shutil
import numpy as np
import json, pickle
from utils.log_manager import get_logger

logger = get_logger(__name__)
# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
#
# # 添加控制台处理器
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
#
# # 设置日志格式
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
#
# # 将处理器添加到日志器
# logger.addHandler(console_handler)

def remove_file(path):
    # 删除目录或文件
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
    except Exception as e:
        print(f"警告：清理失败 {path}, 错误: {e}")

def safe_save_file(original_filename, save_func):
    # save_func(filename)
    # 临时路径和备份路径
    root = os.path.dirname(original_filename)
    name = os.path.basename(original_filename)

    temp_filename = os.path.join(root, f'tmp_{name}')
    backup_filename = os.path.join(root, f'backup_{name}')
    try:
        # 写入临时文件
        save_func(temp_filename)

        # 原子替换
        # 首先删除备份文件
        remove_file(backup_filename)
        if os.path.exists(original_filename):
            os.rename(original_filename, backup_filename)    # 重命名原目录为备份
        os.rename(temp_filename, original_filename)

        # 清洗备份
        remove_file(backup_filename)

        return True
    except Exception as e:
        logger.error(f"错误发生：{str(e)}，启动恢复流程")
        # 如果存在临时文件，则进行替换
        if os.path.exists(temp_filename):
            os.rename(temp_filename, original_filename)
        elif os.path.exists(backup_filename):
            # 没有最新的临时文件，那么使用旧的备份文件替换
            os.rename(backup_filename, original_filename)
        # 清理可能的残留
        remove_file(temp_filename)
        raise RuntimeError(f"数据保存失败，已回滚到之前状态。原始错误: {e}") from e



# json操作
def save_json(data, save_path):
    def default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # ndarray -> list
        elif isinstance(obj, set):
            return list(obj)  # ndarray -> list
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # 对于零维数组或 numpy 标量
            return obj.item()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=default_serializer)
def load_json(load_path, return_type=[]):
    if os.path.exists(load_path):
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    print(f"文件[{load_path}]不存在")
    return return_type

# pickle操作
def save_pickle(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
def load_pickle(load_path, return_type=[]):
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        return data
    print(f"文件[{load_path}]不存在")
    return return_type