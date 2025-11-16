from config import HotpotQAConfig as default_config
from models.memory_rag import GraphMemoryRetriever
from utils.file_manager import load_json, save_json, save_pickle
from utils.evaluation_manager import batch_evaluate_metric_qa_em, batch_calculate_metric_qa_f1, \
    evaluate_metric_extract_match
from utils.dataset_manager import get_musique_samples, get_hotpotqa_samples, get_popqa_samples, get_nq_samples
import argparse
from tqdm import tqdm
import os
import numpy as np
from utils.log_manager import get_logger
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from retrievers.NVEmbedV2 import NVEmbedding
from transformers import AutoTokenizer
from ner.extractor import Extractor
from llms.openai import CacheOpenAI
from llms.vllm_offline import CacheVllm
from utils.class_manager import QuerySolution

logger = get_logger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #

def ablation(data_name,
             task,
         enable_query_decompose=True,
         enable_subquery=True,
         enable_rewrite_query=False,
         enable_cloze=False,
         enable_query_augment=True,
         enable_graph_reranker=True,
         enable_weight_doc_node=True,
         enable_weight_sentence_node=True,
         enable_weight_phrase_node=True,
         enable_weight_query_phrase_node=True):
    default_config.dataset.dataset_name = data_name
    default_config.dataset.dataset_path = f'./datasets/{data_name}/{data_name}.json'
    default_config.dataset.corpus_path = f'./datasets/{data_name}/{data_name}_corpus.json'
    default_config.dataset.output_root = f'./outputs/{data_name}'
    default_config.dataset.log_path = f'{default_config.dataset.output_root}/logs'
    default_config.debug.enable_query_decompose = enable_query_decompose
    default_config.debug.enable_subquery = enable_subquery
    default_config.debug.enable_rewrite_query = enable_rewrite_query
    default_config.debug.enable_cloze = enable_cloze
    default_config.debug.enable_query_augment = enable_query_augment
    default_config.debug.enable_graph_reranker = enable_graph_reranker
    default_config.debug.enable_weight_doc_node = enable_weight_doc_node
    default_config.debug.enable_weight_sentence_node = enable_weight_sentence_node
    default_config.debug.enable_weight_phrase_node = enable_weight_phrase_node
    default_config.debug.enable_weight_query_phrase_node = enable_weight_query_phrase_node
    print(default_config.debug.enable_query_decompose, "????")

    print(default_config.dataset.dataset_name)
    print(default_config.dataset.dataset_path)
    print(default_config.dataset.corpus_path)
    print(default_config.dataset.output_root)
    print(default_config.dataset.log_path)

    tokenizer = AutoTokenizer.from_pretrained(default_config.llm.model_path)

    extractor = Extractor(output_root=default_config.dataset.output_root,
                          max_workers=default_config.rag.max_workers,
                          spacy_model=default_config.extractor.spacy_model,
                          enable_ner_tqdm=default_config.tqdm.enable_ner_tqdm)

    # embedding
    embedding_path = default_config.rag.embedding_path
    max_embedding_length = default_config.rag.max_embedding_length
    model_mode = default_config.rag.model_mode
    if max_embedding_length > 0:
        if model_mode == 'normal':
            embedding_model = HuggingFaceEmbeddings(model_name=embedding_path,
                                                    model_kwargs={"trust_remote_code": True},
                                                    encode_kwargs={'normalize_embeddings': True,
                                                                   'max_length': max_embedding_length,
                                                                   "truncate": True}
                                                    )
        else:
            embedding_model = NVEmbedding(model_name=embedding_path,
                                          # model_kwargs={"trust_remote_code": True},
                                          encode_kwargs={"max_length": max_embedding_length,
                                                         # 32768 from official example,
                                                         "instruction": "",
                                                         "num_workers": 32
                                                         })
    else:
        if model_mode == 'normal':
            embedding_model = HuggingFaceEmbeddings(model_name=embedding_path,
                                                    model_kwargs={"trust_remote_code": True},
                                                    encode_kwargs={' ': True,
                                                                   "truncate": True}
                                                    )
        else:
            embedding_model = NVEmbedding(model_name=embedding_path,
                                          # model_kwargs={"trust_remote_code": True},
                                          encode_kwargs={"max_length": 2048,  # 32768 from official example,
                                                         "instruction": "",
                                                         "num_workers": 32
                                                         })
    print("????", embedding_path, model_mode)
    llm_output_root = os.path.join(default_config.dataset.output_root, 'llm_cache')
    if default_config.llm.enable_online:
        base_llm = CacheOpenAI(api_key=default_config.llm.api_key,
                                    base_url=default_config.llm.base_url,
                                    model=default_config.llm.model,
                                    output_root=llm_output_root,
                                    cache_filename='cache',
                                    cache_table='cache')
    else:
        base_llm = CacheVllm(model_path=default_config.llm.model_path,
                                  output_root=llm_output_root,
                                  cache_filename='cache',
                                  cache_table='cache')

    graph_rag = GraphMemoryRetriever(tokenizer, extractor, embedding_model, base_llm, default_config)

    # 加载数据集
    if default_config.dataset.dataset_name in ['hotpotqa', 'sample100', '2wikimultihopqa']:
        all_queries, all_gold_docs, all_truths = get_hotpotqa_samples(default_config.dataset.dataset_path,
                                                                      default_config.dataset.corpus_path)
    elif default_config.dataset.dataset_name in ['musique']:
        all_queries, all_gold_docs, all_truths = get_musique_samples(default_config.dataset.dataset_path,
                                                                     default_config.dataset.corpus_path)
    elif default_config.dataset.dataset_name in ['popqa']:
        all_queries, all_gold_docs, all_truths = get_popqa_samples(default_config.dataset.dataset_path,
                                                                   default_config.dataset.corpus_path)
    elif default_config.dataset.dataset_name in ['nq_rear']:
        all_queries, all_gold_docs, all_truths = get_nq_samples(default_config.dataset.dataset_path,
                                                                default_config.dataset.corpus_path)
    else:
        all_queries, all_gold_docs, all_truths = get_musique_samples(default_config.dataset.dataset_path,
                                                                     default_config.dataset.corpus_path)

    dataset_size = len(all_queries)
    need_size = dataset_size#int(dataset_size*ratio)
    all_queries = all_queries[:need_size]
    all_gold_docs = all_gold_docs[:need_size]
    all_truths = all_truths[:need_size]
    batch_size = default_config.rag.batch_size

    k_list = [1, 2, 5, default_config.rag.final_search_doc_num]
    all_query_solutions = []
    idx = -1
    for i in tqdm(range(0, len(all_queries), batch_size), desc='检索'):
        idx += 1
        # if idx != 14:
        #     continue
        batch_queries = all_queries[i:i + batch_size]
        batch_gold_docs = all_gold_docs[i:i + batch_size]
        batch_truths = all_truths[i:i + batch_size]
        batch_query_solutions = graph_rag.graph_retrieve(batch_queries, batch_gold_docs=batch_gold_docs, k_list=k_list,
                                                         debug=default_config.debug.enable_debug_recall)
        for query_solution, truth in zip(batch_query_solutions, batch_truths):
            query_solution.gold_answer = truth
            all_query_solutions.append(query_solution)

    # 评估
    epochs = len(all_query_solutions[0].history_iter_prr_reranked_docs)
    pre_ppr_pooled_eval_results_list = [{f"Recall@{k}": 0.0 for k in k_list} for _ in range(epochs)]
    pooled_eval_results_list = [{f"Recall@{k}": 0.0 for k in k_list} for _ in range(epochs)]

    all_pre_qa_solutions = []
    for query_solution in all_query_solutions:
        for epoch in range(epochs):
            for k in k_list:
                pre_ppr_pooled_eval_results_list[epoch][f"Recall@{k}"] += query_solution.pre_ppr_eval_recall[epoch][f"Recall@{k}"]
                pooled_eval_results_list[epoch][f"Recall@{k}"] += query_solution.eval_recall[epoch][f"Recall@{k}"]
            # query_solution.ranked_docs = []#query_solution.ranked_docs[:default_config.rag.final_search_doc_num]
            # query_solution.ranked_doc_keys = []#query_solution.ranked_doc_keys[:default_config.rag.final_search_doc_num]
            # query_solution.ranked_doc_scores = []#query_solution.ranked_doc_scores[:default_config.rag.final_search_doc_num]
            # query_solution.ranked_doc_nodes = []#query_solution.ranked_doc_nodes[:default_config.rag.final_search_doc_num]
            # query_solution.reranked_docs = query_solution.reranked_docs[:default_config.rag.final_search_doc_num]
            # query_solution.reranked_doc_keys = []#query_solution.reranked_doc_keys[:default_config.rag.final_search_doc_num]
            # query_solution.reranked_doc_scores = query_solution.reranked_doc_scores[:default_config.rag.final_search_doc_num]
        all_pre_qa_solutions.append(query_solution.to_dict())
    sample_num = len(all_pre_qa_solutions)

    print(f"===========================DataSet: {data_name}===========================")
    for epoch in range(epochs):
        for k in k_list:
            pre_ppr_pooled_eval_results_list[epoch][f"Recall@{k}"] /= sample_num
            pooled_eval_results_list[epoch][f"Recall@{k}"] /= sample_num
        print(f"=======================Result EPOCH: {epoch}=======================")
        print("Pre PPR Eval: ")
        print(pre_ppr_pooled_eval_results_list[epoch])
        print("Retrieval Eval: ")
        print(pooled_eval_results_list[epoch])

    # 保存信息
    log_info = {'query_solution': all_pre_qa_solutions,
                'pre_ppr_recall': pre_ppr_pooled_eval_results_list,
                'recall': pooled_eval_results_list}

    os.makedirs(default_config.dataset.log_path, exist_ok=True)
    log_name = os.path.basename(default_config.rag.embedding_path)
    log_file = os.path.join(default_config.dataset.log_path, f'eval_{task}_{log_name}_qwen3_embedding_retrieval.pkl')
    save_pickle(log_info, log_file)

    # json_all_query_solutions = []
    # for solution in all_query_solutions:
    #     json_data = solution.to_dict()
    #     json_all_query_solutions.append(json_data)
    # save_json(json_all_query_solutions,
    #           os.path.join(default_config.dataset.log_path, f'eval_{task}_{log_name}_retrieval.json'))

Test_ablation_config = {
    'all': (True, True, False, False, True, True, True, True, True, True),
    'wo_decompose': (False, False, False, False, True, True, True, True, True, True),
    'wo_augment': (True, True, False, False, False, True, True, True, True, True),
    'wo_reranker': (True, True, False, False, True, False, True, True, True, True),
    'wo_doc_node': (True, True, False, False, True, True, False, True, True, True),
    'wo_sentence_node': (True, True, False, False, True, True, True, False, True, True),
    'wo_phrase_node': (True, True, False, False, True, True, True, True, False, True),
    'wo_query_node': (True, True, False, False, True, True, True, True, True, False),
}

Test_ablation_config2 = {
    '+CHG': (False, False, False, False, False, True, True, True, True, True),
    '+CHG+QD': (True, True, False, False, False, True, True, True, True, True),
    'all': (True, True, False, False, True, True, True, True, True, True),
}
ALL_TASK = ['all']#, 'wo_decompose', 'wo_augment', 'wo_reranker', 'wo_doc_node', 'wo_sentence_node', 'wo_phrase_node', 'wo_query_node']
def test_ablation(dataset_name, allowed_test_task=ALL_TASK):
    print(f"{dataset_name}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for task in allowed_test_task:
        ablation(dataset_name, task, *Test_ablation_config2[task])
        print(f"Result: {dataset_name}>>>>>>>>>>>>>{task}")


if __name__ == '__main__':
    # 消融实验
    # for dataset_name in ['hotpotqa','musique', '2wikimultihopqa']:#
    #     test_ablation(dataset_name)
    test_ablation('hotpotqa', ['all'])
    # for dataset_name in ['popqa', '2wikimultihopqa', 'hotpotqa', 'musique', 'nq_rear']:#'musique', '2wikimultihopqa', 'hotpotqa']:
    #     test_ablation(dataset_name, ['+CHG', '+CHG+QD', 'all'])

