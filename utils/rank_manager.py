# 排名算法
from typing import List, Dict
import math
import numpy as np
def standardize_scores(scores: np.ndarray) -> np.ndarray:
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score > min_score:
        scores = (scores - min_score) / (max_score - min_score)
    else:
        scores = np.zeros_like(scores)  # 所有分数相同的情况
    return scores

def standardize_scores_dict(scores):
    """
    对分数进行Z-score标准化
    参数:
        scores (dict): doc -> 分数
    返回:
        dict: doc -> 标准化后的得分
    """
    values = list(scores.values())
    max_value = max(values)
    min_value = min(values)
    return {k: (v - min_value) / (max_value - min_value) for k, v in scores.items()}
def rrf_ranking(sequence_rank_docs, k=60, return_docs=False):
    # Step 1: 构建文档到整数 ID 的映射
    all_docs = set()
    for doc_list in sequence_rank_docs:
        all_docs.update(doc_list)

    doc_to_id = {doc: i for i, doc in enumerate(all_docs)}
    id_to_doc = {i: doc for doc, i in doc_to_id.items()}
    num_docs = len(doc_to_id)

    # Step 2: 转换每个排序列表为 NumPy 数组的整数 ID 形式
    rrf_scores = np.zeros(num_docs, dtype=np.float32)

    for doc_list in sequence_rank_docs:
        ids = np.array([doc_to_id[doc] for doc in doc_list], dtype=np.int32)
        ranks = np.arange(len(doc_list), dtype=np.float32)
        scores = 1.0 / (k + ranks + 1)
        np.add.at(rrf_scores, ids, scores)  # 使用 add.at 支持重复元素累加

    rrf_scores = standardize_scores(rrf_scores)

    # Step 4: 排序并返回 (doc, score)
    sorted_ids = np.argsort(-rrf_scores)
    if return_docs:
        result = [id_to_doc[i] for i in sorted_ids]
    else:
        result = [(id_to_doc[i], float(rrf_scores[i])) for i in sorted_ids]

    return result

def get_rrf_rank(main_rank_docs, sequence_subquery_rank_docs, k=60):
    # 获取子问题文档排名
    # 计算每一个子问题与总问题的rrf排名
    all_subquery_rrf_rank_docs = []
    all_docs = set()
    for subquery_rank_docs in sequence_subquery_rank_docs:
        cur_all_docs = set(subquery_rank_docs) | set(main_rank_docs)
        all_docs.update(cur_all_docs)
        copied = [subquery_rank_docs, main_rank_docs]
        rrf_rank_docs = rrf_ranking(copied, k, return_docs=True)
        all_subquery_rrf_rank_docs.append(rrf_rank_docs)

    # 最后合并
    fusion_rrf_rank_docs = rrf_ranking(all_subquery_rrf_rank_docs + [main_rank_docs])
    return fusion_rrf_rank_docs


def calculate_progress_score(old_scores, new_scores):
    """
    计算进步得分
    参数:
        old_rank (float): 上一次排名
        new_rank (float): 当前排名
    返回:
        float: 进步得分（越高越好）
    """
    doc2progress = {}
    max_progress = 0
    min_progress = float('inf')

    for doc in old_scores:
        old_rank_score = old_scores[doc]
        new_rank_score = new_scores[doc]
        progress = new_rank_score - old_rank_score
        progress = progress if progress > 0 else 0
        if progress < min_progress:
            min_progress = progress
        if progress > max_progress:
            max_progress = progress

        doc2progress[doc] = progress

    for doc in doc2progress:
        doc2progress[doc] = (doc2progress[doc] - min_progress) / (max_progress - min_progress)
        # print(doc2progress[doc])

    return doc2progress

def evaluate_doc_potential2(old_rank_docs: List[str], new_rank_docs: List[str], alpha=0.6, beta=0.4, k=60):
    # List[doc1, doc2...]
    # 构建排名分数：分数越大排名越靠前
    all_docs = set()
    all_docs.update(old_rank_docs)
    all_docs.update(new_rank_docs)


    old_scores = {}
    for rank, doc in enumerate(old_rank_docs):
        old_scores[doc] = 1 / (k + rank + 1)

    new_scores = {}
    for rank, doc in enumerate(new_rank_docs):
        new_scores[doc] = 1 / (k + rank + 1)

    for doc in all_docs:
        if doc not in old_scores:
            old_scores[doc] = 1 / (k + len(old_rank_docs) + 1)

        if doc not in new_scores:
            new_scores[doc] = 1 / (k + len(new_rank_docs) + 1)

    # 标准化分数
    old_scores = standardize_scores_dict(old_scores)
    new_scores = standardize_scores_dict(new_scores)

    # 量化进步
    doc2progress = calculate_progress_score(old_scores, new_scores)

    potential_scores = {}
    new_ranks = sorted(new_scores.keys(), key=lambda x: -new_scores[x])
    for doc, progress_score in doc2progress.items():
        current_score_std = new_scores[doc]
        new_rank = new_ranks.index(doc)
        if doc in new_rank_docs:
            cur_delta = 1/math.sqrt(new_rank_docs.index(doc)+1)
        else:
            cur_delta = 1/math.sqrt(len(new_rank_docs)+1)

        if new_rank <= 100:
            cur_delta *= 2
        potential_score = current_score_std + cur_delta * progress_score
        potential_scores[doc] = potential_score

    return potential_scores

def evaluate_doc_potential(old_rank_docs: List[str], new_rank_docs: List[str], alpha=0.6, beta=0.4, k=60):
    potential_scores1 = evaluate_doc_potential2(old_rank_docs, new_rank_docs, alpha=0.6, beta=0.4, k=k)
    potential_scores2 = evaluate_doc_potential2(new_rank_docs, old_rank_docs, alpha=0.6, beta=0.4, k=k)
    potential_scores = {}
    for doc in potential_scores1:
        score = (potential_scores1[doc] + potential_scores2[doc])/2
        potential_scores[doc] = score
    sorted_docs = sorted(potential_scores.keys(), key=lambda x: -potential_scores[x])
    return sorted_docs

def get_dark_rank(main_rank_docs, sequence_subquery_rank_docs, k=60):
    # 获取子问题文档排名
    all_subquery_rank_docs = []
    # 计算每一个子问题与总问题的rrf排名
    all_subquery_rrf_rank_docs = []
    for subquery_rank_docs in sequence_subquery_rank_docs:
        potential_rank_docs = evaluate_doc_potential(main_rank_docs, subquery_rank_docs, alpha=1, beta=1, k=60)
        rrf_rank_docs = potential_rank_docs
        all_subquery_rrf_rank_docs.append(rrf_rank_docs)
        all_subquery_rank_docs.append(subquery_rank_docs)

    # 最后合并
    all_rank_docs = []
    for idx, (sub_rank_docs, rrf_rank_docs) in enumerate(zip(all_subquery_rank_docs, all_subquery_rrf_rank_docs)):
        potential_rank_docs1 = evaluate_doc_potential(sub_rank_docs, rrf_rank_docs, alpha=1, beta=1, k=60)
        # potential_rank_docs2 = evaluate_doc_potential(main_rank_docs, rrf_rank_docs, alpha=1, beta=1, k=60)
        # fusion_rrf_rank_docs= rrf_ranking([potential_rank_docs1, potential_rank_docs2], k, return_docs=True)

        all_rank_docs.append(potential_rank_docs1)

    # 最后合并
    fusion_rrf_rank_docs = rrf_ranking(all_rank_docs + [main_rank_docs], k)
    return fusion_rrf_rank_docs
