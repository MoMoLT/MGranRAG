from collections import defaultdict
import numpy as np
# 并查集思想
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x


def merge_equivalent_phrases(graph_equivalent_phrases):
    uf = UnionFind()

    # Step 1: 构建实体之间的关系（如果集合之间有交集就合并）
    phrases = list(graph_equivalent_phrases.keys())

    for i, a in enumerate(phrases):
        for b in phrases[i + 1:]:
            if graph_equivalent_phrases[a] & graph_equivalent_phrases[b]:
                uf.union(a, b)
    # Step 2: 收集所有连通组件
    components = defaultdict(set)
    for phrase in phrases:
        root = uf.find(phrase)
        components[root].add(phrase)

    # Step 3: 构造 new_graph_equivalent_phrases
    new_graph_equivalent_phrases = dict()

    for comp in components.values():
        # 选一个 leader（可替换为你自己的策略，如最长字符串、频率最高等）
        leader = sorted(comp)[0]#random.choice(list(comp))
        # 合并所有成员的集合
        merged_set = set()
        for phrase in comp:
            merged_set.update(graph_equivalent_phrases.get(phrase, set()))
            merged_set.add(phrase)  # 可选：包含 phrase 自身

        new_graph_equivalent_phrases[leader] = merged_set

    return new_graph_equivalent_phrases


def compute_rrf_scores(scores: np.ndarray, k=60):
    # 创建一个与 scores 形状相同的数组，用于存放每个元素对应的 rank（从高到低）
    ranks = np.empty_like(scores, dtype=int)
    # 获取从高到低的排序索引，并依次分配 rank: 最高分 rank=0, 第二高 rank=1...
    ranks[np.argsort(-scores)] = np.arange(len(scores))

    # 计算 RRF 分数
    rrf_scores = 1 / (k + ranks + 1)
    return rrf_scores

def compute_positive_rrf_scores(scores: np.ndarray, k=60):
    scores = np.asarray(scores)
    zeros_mask = (scores == 0)

    # 只对非零分数进行排名
    non_zero_scores = scores[~zeros_mask]
    if len(non_zero_scores) == 0:
        return np.zeros_like(scores, dtype=float)  # 全是 0 的情况直接返回全 0

    non_zero_indices = np.argsort(-non_zero_scores)
    non_zero_ranks = np.empty_like(non_zero_scores, dtype=int)
    non_zero_ranks[non_zero_indices] = np.arange(len(non_zero_scores))

    # 创建完整 rank 数组
    full_ranks = np.empty_like(scores, dtype=int)
    full_ranks[~zeros_mask] = non_zero_ranks
    max_rank = len(non_zero_scores) - 1
    full_ranks[zeros_mask] = max_rank + 1  # 0 的 rank 排到最后

    # 计算 RRF 分数
    rrf_scores = 1 / (k + full_ranks + 1)

    # 将原本为 0 的位置也设置为 0（关键改动）
    rrf_scores[zeros_mask] = 0

    return rrf_scores