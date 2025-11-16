import numpy as np
from scipy.stats import rankdata

def zscore_normalize_weights(weights):
    """
    对字典中的权重进行 Z-Score 标准化
    :param weights: dict, 如 {'a': 1.2, 'b': 3.5, ...}
    :return: dict, 标准化后的 z-scores，如 {'a': -0.5, 'b': 1.2, ...}
    """
    if not weights:
        return {}

    values = np.array(list(weights.values()))
    mean = np.mean(values)
    std = np.std(values)

    # 防止除以 0（当所有值相等时）
    if std == 0:
        z_scores = np.zeros_like(values)
    else:
        z_scores = (values - mean) / std

    return {k: float(z_scores[i]) for i, k in enumerate(weights.keys())}

def zscore_normalize_weights_non_negative(weights):
    """
    对字典中的权重进行 Z-Score 标准化，并保留非负值
    :param weights: dict, 如 {'a': 1.2, 'b': 3.5, ...}
    :return: dict, 只包含 Z-Score ≥ 0 的项
    """
    if not weights:
        return {}

    values = np.array(list(weights.values()))
    mean = np.mean(values)
    std = np.std(values)

    # 防止除以 0（当所有值相等时）
    if std == 0:
        z_scores = np.zeros_like(values, dtype=np.float64)
    else:
        z_scores = (values - mean) / std

    # 只保留非负值
    non_negative_mask = z_scores >= 0
    non_negative_zscores = z_scores[non_negative_mask]
    corresponding_keys = [k for k, keep in zip(weights.keys(), non_negative_mask) if keep]

    return {k: float(non_negative_zscores[i]) for i, k in enumerate(corresponding_keys)}

def weights_to_ranks(weights_groups):
    """
    将多组权重转换为排名列表（降序，排名从1开始）
    :param weights_groups: List[np.array] - 多组权重数组，每组形状为 (node_nums,)
    :return: List[np.array] - 每组对应的排名列表
    """
    if not weights_groups:
        raise ValueError("权重组不能为空")

    node_nums = weights_groups[0].shape[0]
    rank_groups = []

    for weights in weights_groups:
        if len(weights) != node_nums:
            raise ValueError("所有权重数组长度必须一致")
        # 使用 scipy 的 rankdata 进行密集排名（相同权重共享排名）
        ranks = rankdata(-weights, method='dense')  # 降序排名
        rank_groups.append(ranks)

    return rank_groups


def rrf_fusion(rank_groups, k=60):
    """
    RRF 分数融合
    :param rank_groups: List[np.array] - 多组排名列表
    :param k: int - 平滑因子（默认60）
    :return: np.array - 融合后的 RRF 分数
    """
    if not rank_groups:
        raise ValueError("排名组不能为空")

    # 向量化操作：将 rank_groups 转换为 NumPy 数组
    rank_array = np.array(rank_groups)  # shape: (M, N)
    # 计算 RRF 分数（向量化）
    rrf_scores = np.sum(1.0 / (k + rank_array), axis=0)  # shape: (N,)
    return rrf_scores

def normalize_scores(scores):
    """
    最大-最小归一化
    :param scores: np.array - RRF 分数
    :return: np.array - 归一化后的权重
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    if np.isclose(max_score, min_score):
        return np.zeros_like(scores)
    normalized = (scores - min_score) / (max_score - min_score)
    return normalized

def fusion_graph_weights(weights_groups):
    # Step 1: 转换为排名列表
    rank_groups = weights_to_ranks(weights_groups)
    # Step 2: RRF 融合
    rrf_scores = rrf_fusion(rank_groups, k=60)
    # Step 3: 归一化到 [0,1]
    final_weights = normalize_scores(rrf_scores)
    return final_weights
