'''
为了方便缓存管理，每一种数据结构，一个sqalite. 因为随着文件的扩大，读取速度会降低，有些数据会比较大，而有些数据操作比较小。
'''
import json
import sqlite3
from filelock import FileLock
from typing import List, Dict
import os
import pandas as pd
# 保存缓存

def save_cache(cache_root: str, sequences: List[Dict], filename: str='cache', table_name: str='cache'):
    # [Dict]
    cache_filename = os.path.join(cache_root, f'{filename}.sqlite')
    data_to_insert = []
    for item in sequences:
        key = item['key']
        metadata = json.dumps(item, sort_keys=True, default=str)
        data_to_insert.append((key, metadata))

    lock_file = cache_filename + ".lock"
    with FileLock(lock_file):
        conn = sqlite3.connect(cache_filename)
        c = conn.cursor()
        # 确保表存在
        c.execute(f"""CREATE TABLE IF NOT EXISTS {table_name} (
        key TEXT PRIMARY KEY,
        metadata TEXT
        )""")
        c.executemany(
            f"INSERT OR REPLACE INTO {table_name} (key, metadata) VALUES (?, ?)",
            data_to_insert
        )
        conn.commit()
        conn.close()

# 通过key读取缓存
def load_cache(cache_root: str, sequence_key: List[str], filename: str='cache', table_name: str='cache'):
    cache_filename = os.path.join(cache_root, f'{filename}.sqlite')
    lock_file = cache_filename + ".lock"
    with FileLock(lock_file):
        conn = sqlite3.connect(cache_filename)
        c = conn.cursor()
        c.execute(f"""CREATE TABLE IF NOT EXISTS {table_name} (
        key TEXT PRIMARY KEY,
        metadata TEXT
        )""")
        conn.commit()  # commit to save the table creation
        placeholders = ','.join(['?'] * len(sequence_key))  # 生成占位符
        c.execute(f"""SELECT key, metadata 
        FROM {table_name}
        WHERE key IN ({placeholders})
        """, sequence_key)

        rows = c.fetchall()  # 获取所有匹配结果
        conn.close()
    results = {}
    for row in rows:
        key, metadata_str = row
        metadata = json.loads(metadata_str)
        results[key] = metadata
    sequence_result = []
    for key in sequence_key:
        sequence_result.append(results[key])
    return sequence_result


# 获取所有key值
def get_cache_keys(cache_root: str, filename: str='cache', table_name: str='cache'):
    cache_filename = os.path.join(cache_root, f'{filename}.sqlite')
    lock_file = cache_filename + ".lock"
    with FileLock(lock_file):
        conn = sqlite3.connect(cache_filename)
        c = conn.cursor()
        # if the table does not exist, create it
        c.execute(f"""CREATE TABLE IF NOT EXISTS {table_name} (
        key TEXT PRIMARY KEY,
        metadata TEXT
        )""")
        conn.commit()  # commit to save the table creation

        c.execute(f"SELECT key FROM {table_name}")  # 查询所有key列的值
        keys = [row[0] for row in c.fetchall()]  # 提取结果中的第一列
        conn.close()
    return keys

def save_parquet(cache_root: str, dict_data: Dict, filename: str):
    df = pd.DataFrame(dict_data)
    path = os.path.join(cache_root, f'{filename}.parquet')
    df.to_parquet(path, engine='pyarrow')

def load_parquet(cache_root: str, filename: str, columns: List[str]=None):
    path = os.path.join(cache_root, f'{filename}.parquet')
    if columns is None:
        df = pd.read_parquet(path, engine='pyarrow')
    else:
        df = pd.read_parquet(path, engine='pyarrow', columns=columns)

    restored_dict = df.to_dict(orient='list')
    return restored_dict