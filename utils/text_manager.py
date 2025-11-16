import re
import hashlib
from rapidfuzz import process, fuzz
from typing import List, Dict
import numpy as np
# 生成chunk的key
def generate_key(text: str, prefix: str = "") -> str:
    return prefix + hashlib.sha256(text.encode()).hexdigest()
def has_uppercase(text):
    """使用正则表达式检查大写字母"""
    return bool(re.search(r'[A-Z]', text))
def has_number(text):
    """使用正则表达式检查文本中是否包含数值"""
    return bool(re.search(r'\d+', text))

def pick_numbers(text):
    numbers = re.findall(r'\d+', text)

    return numbers  # 输出: ['1971', '192']

def check_numbers_relations(search, check_text):
    # 检测两个之间是否存在数值偏差
    search_numbers = set(pick_numbers(search))
    if len(search_numbers) == 0:
        return True
    check_text_numbers = set(pick_numbers(check_text))
    if len(check_text_numbers) == 0:
        return True
    if len(check_text_numbers & search_numbers) > 0:
        return True

    return False

def strip_query(query):
    entities = re.findall('\[(.*?)\]', query, re.S)
    for e in entities:
        if has_uppercase(e):
            query = query.replace(f'[{e}]', e)
        else:
            query = query.replace(f'[{e}]', f'[{e}?]')
    return query

# 模糊相似度
def batch_get_similar_keywords2(
        queries: List[str],
        dictionary: List[str],
        cutoff: float = 70.0,
        n: int = 3
) -> Dict[str, List[str]]:
    """
    批量获取相似关键词 (优化版)

    参数:
        queries: 待查询词列表
        dictionary: 目标词典
        threshold: 相似度阈值(0-100)

    返回:
        {查询词: [相似词1, 相似词2,...]}
    """
    # 转换为numpy数组提升性能
    dict_array = np.array(dictionary)

    # 批量计算相似度矩阵
    similarity_matrix = process.cdist(
        queries,
        dictionary,
        scorer=fuzz.ratio,
        workers=-1
    )

    result_dict = {}
    print(queries)
    print(dictionary)
    print(">>>><<<>>><<>M><><><><><")
    print(similarity_matrix.max())
    for i, query in enumerate(queries):
        # 使用NumPy高效筛选 (比列表推导快5-10倍)
        mask = similarity_matrix[i] >= cutoff
        matched_words = dict_array[mask]

        # 按相似度降序排序
        matched_scores = similarity_matrix[i][mask]
        sorted_words = [
            word for _, word in sorted(
                zip(matched_scores, matched_words)[:n],
                reverse=True
            )
        ]

        result_dict[query] = sorted_words

    return result_dict

def batch_get_similar_keywords3(
    queries: List[str],
    dictionary: List[str],
    cutoff: float = 80.0,
    n: int = 3
) -> Dict[str, List[str]]:
    """
    批量获取相似关键词 (优化版)

    参数:
        queries: 待查询词列表
        dictionary: 目标词典
        threshold: 相似度阈值(0-100)
        n: 最多返回的匹配词数量

    返回:
        {查询词: [相似词1, 相似词2,...]}
    """
    # 将词典转换为 NumPy 数组，提高索引效率
    dict_array = np.array(dictionary)
    print(queries)
    print(dictionary)
    print(">>>><<<>>><<>M><><><><><")

    # 批量计算相似度矩阵，使用 fuzz.ratio 作为相似度算法
    similarity_matrix = process.cdist(
        queries,
        dictionary,
        scorer=fuzz.ratio,
        # workers=-1  # 利用所有 CPU 核心加速
    )

    result_dict = {}
    print(similarity_matrix.max())
    for i, query in enumerate(queries):
        # 使用 NumPy 高效筛选 (比列表推导快5-10倍)
        mask = similarity_matrix[i] >= cutoff
        matched_words = dict_array[mask]

        # 提取对应的相似度得分
        matched_scores = similarity_matrix[i][mask]

        # 按相似度降序排序，并取前 n 个
        sorted_pairs = sorted(zip(matched_scores, matched_words), reverse=True)
        sorted_words = [word for _, word in sorted_pairs[:n]]

        result_dict[query] = sorted_words

    return result_dict


import numpy as np
from rapidfuzz import process, fuzz
from typing import List, Dict

def batch_get_similar_keywords(
        queries: List[str],
        dictionary: List[str],
        cutoff: float = 70.0,
        n: int = 3
) -> Dict[str, List[str]]:
    """
    批量获取相似关键词 (优化加速版)

    参数:
        queries: 待查询词列表
        dictionary: 目标词典
        cutoff: 相似度阈值(0-100)
        n: 返回每个查询词的最大结果数量

    返回:
        {查询词: [相似词1, 相似词2,...]} (按相似度降序排列，最多n个结果)
    """
    result_dict = {}
    # 批量计算相似度矩阵 (直接获取TopN结果)
    for query in queries:
        # 使用process.extract直接获取TopN结果，避免全量计算
        matches = process.extract(
            query,
            dictionary,
            scorer=fuzz.ratio,
            score_cutoff=cutoff,
            limit=n
        )
        # 提取词并按分数降序排列
        # result_dict[query] = [word for word, score, _ in sorted(matches, key=lambda x: -x[1])]
        result_dict[query] = [word for word, score, _ in matches]

    return result_dict