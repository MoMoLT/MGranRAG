from config import HotpotQAConfig as default_config
from models.memory_rag import MGranRAG
from utils.file_manager import load_pickle, save_json, save_pickle
from utils.evaluation_manager import evaluate_metric_recall_k, batch_evaluate_metric_qa_em, batch_calculate_metric_qa_f1, evaluate_metric_extract_match
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
import copy
logger = get_logger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #

def get_embedding_model(config):
    if config.rag.max_embedding_length > 0:
        if config.rag.model_mode == 'normal':
            embedding_model = HuggingFaceEmbeddings(model_name=config.rag.embedding_path,
                                                    model_kwargs={"trust_remote_code": True},
                                                    encode_kwargs={'normalize_embeddings': True}
                                                    )
        else:
            embedding_model = NVEmbedding(model_name=config.rag.embedding_path,
                                          encode_kwargs={"max_length": config.rag.max_embedding_length,
                                                         "instruction": "",
                                                         "num_workers": 32
                                                         })
    else:
        if config.rag.model_mode == 'normal':
            embedding_model = HuggingFaceEmbeddings(model_name=config.rag.embedding_path,
                                                    model_kwargs={"trust_remote_code": True},
                                                    encode_kwargs={'normalize_embeddings': Truee}
                                                    )
        else:
            embedding_model = NVEmbedding(model_name=config.rag.embedding_path,
                                          encode_kwargs={"max_length": 2048,  # 32768 from official example,
                                                         "instruction": "",
                                                         "num_workers": 32
                                                         })
    return embedding_model

def get_llm(config):
    llm_output_root = os.path.join(config.dataset.output_root, 'llm_cache')
    if config.llm.enable_online:
        base_llm = CacheOpenAI(api_key=config.llm.api_key,
                               base_url=config.llm.base_url,
                               model=config.llm.model,
                               output_root=llm_output_root,
                               cache_filename='cache',
                               cache_table='cache')
    else:
        base_llm = CacheVllm(model_path=config.llm.model_path,
                             output_root=llm_output_root,
                             cache_filename='cache',
                             cache_table='cache')
    return base_llm


def rag_retrieve(graph_rag, all_queries, batch_size):
    all_query_solutions = []
    for i in tqdm(range(0, len(all_queries), batch_size), desc='检索'):
        batch_queries = all_queries[i:i + batch_size]
        batch_query_solutions = graph_rag.graph_retrieve(batch_queries)
        all_query_solutions.extend(batch_query_solutions)
    return all_query_solutions

def rag_qa(graph_rag, all_query_solutions, batch_size):
    new_all_query_solutions = []
    for i in tqdm(range(0, len(all_query_solutions), batch_size), desc='检索'):
        batch_query_solutions = all_query_solutions[i:i + batch_size]
        batch_query_solutions = graph_rag.graph_qa(batch_query_solutions)
        new_all_query_solutions.extend(batch_query_solutions)
    return new_all_query_solutions


def qa(graph_rag, all_query_solutions, batch_size):
    all_query_solutions = rag_qa(graph_rag, all_query_solutions, batch_size)

    all_gold_answers = []
    all_responses = []
    new_all_query_solutions = []
    for i in range(len(all_query_solutions)):
        query_solution = all_query_solutions[i]
        query_solution.eval_em = evaluate_metric_extract_match(query_solution.gold_answer, query_solution.response)
        all_gold_answers.append(query_solution.gold_answer)
        all_responses.append(query_solution.response)
        new_all_query_solutions.append(query_solution.to_dict())

    overall_qa_em_result, example_qa_em_results = batch_evaluate_metric_qa_em(gold_answers=all_gold_answers,
                                                                              predicted_answers=all_responses,
                                                                              aggregation_fn=np.max)
    overall_qa_f1_result, example_qa_f1_results = batch_calculate_metric_qa_f1(gold_answers=all_gold_answers,
                                                                               predicted_answers=all_responses,
                                                                               aggregation_fn=np.max)
    overall_qa_em_result.update(overall_qa_f1_result)
    overall_qa_results = overall_qa_em_result
    overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}

    return new_all_query_solutions, overall_qa_results

def ablation(data_name,
             filter_doc_num=10,
             search_epochs=5,
             enable_query_decompose=True,
             enable_subquery=True,
             enable_rewrite_query=False,
             enable_cloze=False,
             enable_query_augment=True,
             enable_graph_reranker=True,
             enable_weight_doc_node=True,
             enable_weight_sentence_node=True,
             enable_weight_phrase_node=True,
             enable_weight_query_phrase_node=True,
             mode='retrieve',
             url=None,
             model_name=None,
             model_path=None):

    config = copy.deepcopy(default_config)
    config.dataset.dataset_name = data_name
    config.dataset.dataset_path = f'./datasets/{data_name}/{data_name}.json'
    config.dataset.corpus_path = f'./datasets/{data_name}/{data_name}_corpus.json'
    config.dataset.output_root = f'./outputs/{data_name}'
    config.dataset.log_path = f'{config.dataset.output_root}/logs'

    config.rag.filter_doc_num = filter_doc_num
    config.rag.search_epochs = search_epochs
    config.debug.enable_query_decompose = enable_query_decompose
    config.debug.enable_subquery = enable_subquery
    config.debug.enable_rewrite_query = enable_rewrite_query
    config.debug.enable_cloze = enable_cloze
    config.debug.enable_query_augment = enable_query_augment
    config.debug.enable_graph_reranker = enable_graph_reranker
    config.debug.enable_weight_doc_node = enable_weight_doc_node
    config.debug.enable_weight_sentence_node = enable_weight_sentence_node
    config.debug.enable_weight_phrase_node = enable_weight_phrase_node
    config.debug.enable_weight_query_phrase_node = enable_weight_query_phrase_node

    if url is not None:
        config.llm.base_url = url
        if model_name is not None:
            config.llm.model = model_name
        if model_path is not None:
            config.llm.model_path = model_path

    # Load Dataset
    if config.dataset.dataset_name in ['hotpotqa', 'sample100', '2wikimultihopqa']:
        all_queries, all_gold_docs, all_truths = get_hotpotqa_samples(config.dataset.dataset_path,
                                                                      config.dataset.corpus_path)
    elif config.dataset.dataset_name in ['musique']:
        all_queries, all_gold_docs, all_truths = get_musique_samples(config.dataset.dataset_path,
                                                                     config.dataset.corpus_path)
    elif config.dataset.dataset_name in ['popqa']:
        all_queries, all_gold_docs, all_truths = get_popqa_samples(config.dataset.dataset_path,
                                                                   config.dataset.corpus_path)
    elif config.dataset.dataset_name in ['nq_rear']:
        all_queries, all_gold_docs, all_truths = get_nq_samples(config.dataset.dataset_path,
                                                                config.dataset.corpus_path)
    else:
        all_queries, all_gold_docs, all_truths = get_musique_samples(config.dataset.dataset_path,
                                                                     config.dataset.corpus_path)

    tokenizer = AutoTokenizer.from_pretrained(config.llm.model_path)

    extractor = Extractor(output_root=config.dataset.output_root,
                          max_workers=config.rag.max_workers,
                          spacy_model=config.extractor.spacy_model,
                          enable_ner_tqdm=config.tqdm.enable_ner_tqdm)

    embedding_model = get_embedding_model(config)
    base_llm = get_llm(config)
    graph_rag = MGranRAG(tokenizer, extractor, embedding_model, base_llm, config)
    batch_size = config.rag.batch_size

    if mode in ['retrieval', 'end2end']:
        all_query_solutions = rag_retrieve(graph_rag, all_queries, batch_size)
        k_list = [1, 2, 5, config.rag.final_search_doc_num]
        # 开始评估
        for i in range(len(all_queries)):
            solution = all_query_solutions[i]
            gold_docs = all_gold_docs[i]
            truth = all_truths[i]

            solution.gold_docs = gold_docs
            solution.gold_answer = truth

            # PPR之前的
            pre_ppr_recall_list = []
            for reranked_docs in solution.history_iter_pre_ppr_reranked_docs:
                recall_dict = evaluate_metric_recall_k(gold_docs, reranked_docs, k_list)
                pre_ppr_recall_list.append(recall_dict)

            final_recall_list = []
            for reranked_docs in solution.history_iter_prr_reranked_docs:
                recall_dict = evaluate_metric_recall_k(gold_docs, reranked_docs, k_list)
                final_recall_list.append(recall_dict)

            solution.pre_ppr_eval_recall = pre_ppr_recall_list
            solution.eval_recall = final_recall_list

        # 综合评估
        epochs = len(all_query_solutions[0].history_iter_prr_reranked_docs)
        pre_ppr_pooled_eval_results_list = [{f"Recall@{k}": 0.0 for k in k_list} for _ in range(epochs)]
        pooled_eval_results_list = [{f"Recall@{k}": 0.0 for k in k_list} for _ in range(epochs)]
        for query_solution in all_query_solutions:
            for epoch in range(epochs):
                for k in k_list:
                    pre_ppr_pooled_eval_results_list[epoch][f"Recall@{k}"] += query_solution.pre_ppr_eval_recall[epoch][f"Recall@{k}"]
                    pooled_eval_results_list[epoch][f"Recall@{k}"] += query_solution.eval_recall[epoch][f"Recall@{k}"]

        sample_num = len(all_query_solutions)
        print(f"===========================DataSet: {data_name}===========================")
        for epoch in range(epochs):
            for k in k_list:
                pre_ppr_pooled_eval_results_list[epoch][f"Recall@{k}"] /= sample_num
                pooled_eval_results_list[epoch][f"Recall@{k}"] /= sample_num
            print(f"=======================Round {epoch} Recall=======================")
            print("Pre PPR Eval: ")
            print(pre_ppr_pooled_eval_results_list[epoch])
            print("Retrieval Eval: ")
            print(pooled_eval_results_list[epoch])

        if mode == 'retrieval':
            log_info = {'query_solution': all_query_solutions,
                        'pre_ppr_recall': pre_ppr_pooled_eval_results_list,
                        'recall': pooled_eval_results_list}

            os.makedirs(default_config.dataset.log_path, exist_ok=True)
            log_file = os.path.join(default_config.dataset.log_path, f'all_query_solutions.pkl')
            save_pickle(log_info, log_file)
    elif mode == 'qa':
        log_file = os.path.join(default_config.dataset.log_path, f'all_query_solutions.pkl')
        log_info = load_pickle(log_file)
        all_query_solutions = log_info['query_solution']
        pre_ppr_pooled_eval_results_list = log_info['pre_ppr_recall']
        pooled_eval_results_list = log_info['recall']
    else:
        raise ValueError("Mode is Error: retrieval/qa/end2end")
        return

    if mode in ['end2end', 'qa']:
        new_all_query_solutions, overall_qa_results = qa(graph_rag, all_query_solutions, batch_size)
        print(f"===========================DataSet: {data_name}===========================")
        print(f"Evaluation results for QA: {overall_qa_results}")

        # 保存信息
        log_info = {'query_solution': new_all_query_solutions,
                    'pre_ppr_recall': pre_ppr_pooled_eval_results_list,
                    'recall': pooled_eval_results_list,
                    'qa': overall_qa_results}

        os.makedirs(default_config.dataset.log_path, exist_ok=True)
        log_file = os.path.join(default_config.dataset.log_path, f'eval_retrieval_qa.json')
        save_json(log_info, log_file)



def main():
    parser = argparse.ArgumentParser(description="MGranRAG")
    # parser.add_argument('--config', type=str, default='test_embedding', help='config name')
    parser.add_argument('--data', type=str, default='sample100', help='dataset name')
    parser.add_argument('--filter_doc_num', type=int, default=10, help='filter passages num')
    parser.add_argument('--epoch', type=int, default=1, help='retrieval epoch')
    parser.add_argument('--mode', type=str, default='retrieval', help='RAG mode: retrieval/qa/end2end')
    parser.add_argument('--url', type=str, default=None, help='llm url')
    parser.add_argument('--model_name', type=str, default=None, help='llm model_name')
    parser.add_argument('--model_path', type=str, default=None, help='llm mode_path')

    args = parser.parse_args()

    ablation(args.data,
             filter_doc_num=args.filter_doc_num,
             search_epochs=args.epoch,
             enable_query_decompose=True,
             enable_subquery=True,
             enable_rewrite_query=False,
             enable_cloze=False,
             enable_query_augment=True,
             enable_graph_reranker=True,
             enable_weight_doc_node=True,
             enable_weight_sentence_node=True,
             enable_weight_phrase_node=True,
             enable_weight_query_phrase_node=True,
             mode=args.mode,
             url=args.url,
             model_name=args.model_name,
             model_path=args.model_path)

if __name__ == '__main__':
    main()
