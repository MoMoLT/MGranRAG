from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np

Triple = Tuple[str, str, str]

@dataclass
class QuerySolution:
    # 静态数据
    query: str
    subqueries: List[str] = None
    rewritten_query: str = None
    query_phrases: set = None


    # 动态参数-临时缓存
    augment_query: str = None                    # 这是给retriever检索的query内容
    ranked_docs: List[str] = None
    ranked_doc_keys: List[str] = None
    ranked_doc_scores: np.ndarray = None
    ranked_doc_nodes: List[int] = None

    reranked_docs: List[str] = None  # 最终检索的文档
    reranked_doc_keys: List[str] = None  # 最终检索的文档
    reranked_doc_scores: np.ndarray = None

    history_iter_pre_ppr_reranked_docs: List[List[str]] = None  # 记录每一次查询增强的结果
    history_iter_prr_reranked_docs: List[List[str]] = None  # 记录每一轮Graph重排序结果(200)个
    # unprocessed_subqueries: List[str] = None    # 判断哪些子问题还需要证据支撑, query和rewritten_query每次迭代都会存在，只有subqueries不断删除
    # 这是search_query检索的结果, 默认取前search_top*2个文档

    evidence_note: Dict = None                  # 筛选后的所有证据 {sentence_node: {id:node, content, org_content, keywords}}
    evidence_update_status: bool = None         # evidence的更新状态
    filter_responses: List[str] = None          # 记录filter的回答
    processed_document_nodes: set = None        # 记录所有处理过的doc node

    fully_supported_sentence_nodes: set = None  # 筛选后的fully级句子
    cur_iter_sentence_nodes: set = None
    # shallow_memory上所有与query相关的phrase节点

    evidence_entity_phrase_nodes: np.ndarray = None
    evidence_entity_phrase_node_scores: np.ndarray = None
    evidence_concept_phrase_nodes: np.ndarray = None
    evidence_concept_phrase_node_scores: np.ndarray = None
    unevidence_entity_phrase_nodes: np.ndarray = None
    unevidence_entity_phrase_node_scores: np.ndarray = None

    graph_phrase_scores: np.ndarray = None
    graph_phrase_factors: np.ndarray = None

    query_entity_phrase_nodes: Dict = None
    query_concept_phrase_nodes: Dict = None
    # shallow_memory上所有与evidence_phrase相关的phrase节点
    fully_supported_evidence_phrase_nodes: Dict = None
    partially_supported_evidence_phrase_nodes: Dict = None
    unfiltered_fully_supported_evidence_phrase_nodes: Dict = None
    unfiltered_partially_supported_evidence_phrase_nodes: Dict = None

    document_ratio: float = None
    sentence_ratio: float = None
    phrase_ratio: float = None
    entity_phrase_factor: float = None
    concept_phrase_factor: float = None
    query_phrase_factor: float = None
    sentence_phrase_factor: float = None
    filtered_phrase_factor: float = None
    unfiltered_phrase_factor: float = None

    global_graph_scores: np.ndarray = None
    ppr_graph_scores: np.ndarray = None
    prediction_status: bool = None  # 是否对query有响应了
    response: str = None  # LLM对query的响应

    gold_docs: List[str] = None
    gold_answer: set = None

    pre_ppr_eval_recall: List[List[Dict]] = None
    eval_recall: List[List[Dict]] = None
    eval_em: float = None


    def to_dict(self):
        return {"question": self.query,
                "subqueries": list(self.subqueries),
                "rewritten_query": self.rewritten_query,
                "augment_query": self.augment_query,
                "query_phrases": list(self.query_phrases),
                "evidence_note": [{'node': node, 'content': item['content'], 'keywords': list(item['keywords']), "relevant_clozes": list(item['relevant_clozes']), 'score': item['score']} for node, item in
                                  self.evidence_note.items()],
                "filter_responses": list(self.filter_responses),
                "pre_ppr_search_docs": [list(ranked_docs[:10]) for ranked_docs in self.history_iter_pre_ppr_reranked_docs],
                "search_docs": [list(ranked_docs[:10]) for ranked_docs in self.history_iter_prr_reranked_docs],
                "gold_docs": list(self.gold_docs),
                "gold_answer": list(self.gold_answer),
                "fully_supported_sentence_nodes": list(self.fully_supported_sentence_nodes),
                "pre_ppr_eval_recall": self.pre_ppr_eval_recall,
                "eval_recall": self.eval_recall,
                "unevidence_entity_phrases": self.unevidence_entity_phrases,
                "evidence_entity_phrases": self.evidence_entity_phrases,
                "evidence_concept_phrases": self.evidence_concept_phrases,
                "query_entity_phrase_nodes": self.query_entity_phrase_nodes,
                "query_concept_phrase_nodes": self.query_concept_phrase_nodes
               }


    def to_qa_solution(self):
        return QuerySolution(query=self.query,
                             subqueries=self.subqueries,
                             rewritten_query=self.rewritten_query,
                             query_phrases=self.query_phrases,
                             history_iter_pre_ppr_reranked_docs=self.history_iter_pre_ppr_reranked_docs,
                             history_iter_prr_reranked_docs=self.history_iter_prr_reranked_docs,
                             evidence_note=self.evidence_note,
                             fully_supported_sentence_nodes=self.fully_supported_sentence_nodes,
                             gold_docs=self.gold_docs,
                             gold_answer=self.gold_answer,
                             reranked_docs=self.reranked_docs[:10],
                             )