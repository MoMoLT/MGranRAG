'''检索器
功能简介：
1. 语料库向量化：可以指定语料库进行向量化。对文档进行切块：句子级，片段级， 还有实体级
2. 批量检索问题：根据一个问题，有两种检索： <1>从全局文件中检索, <2>给定文件范围检索（即给定范围内所有文档的Keys）
3. 缓存机制：
'''

import os
import json
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict
from utils.text_manager import generate_key
from utils.cache_manager import save_cache, load_cache, get_cache_keys
from concurrent.futures import ThreadPoolExecutor
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from retrievers.NVEmbedV2 import NVEmbedding
from retrievers.faiss_embedding_manager import ChunkStore
from utils.log_manager import get_logger
logger = get_logger(__name__)

'''
目录结构
- output_root:
    - graph_rag_model
    - extractor
    - retriever
        - embedding_name
            - vector_store
            - cache
    - info_filter
        - LLM_name
    - decomposer
        - LLM_name
'''

def get_query_instruction(search_mode):
    instructions = {
        'phrase_to_phrase': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_doc': 'Given a question, retrieve relevant documents that best answer the question.',
        'query_to_sentence': 'Given a question, retrieve relevant sentences that best answer the question.',
        'agument_query_to_doc': 'Given a multi-hop question and some relevant knowledges, retrieve relevant documents that can help answer the complex question.',
    }
    default_instruction = instructions['query_to_doc']
    return instructions.get(search_mode, default_instruction)

class CacheRetriever:
    def __init__(self, embedding_model, embedding_path: str, output_root: str, max_workers: int=100, model_mode='normal', enable_search_tqdm=False):
        self.embedding_name = os.path.basename(embedding_path)
        # 加载embedding_model
        self.embedding_model = embedding_model

        self.max_workers = max_workers
        self.model_mode = model_mode
        self.enable_search_tqdm = enable_search_tqdm

        # 根目录位置
        self.root = os.path.join(output_root, "retriever", self.embedding_name)
        if os.path.exists(self.root):
            self.exist_retriever = True
        else:
            self.exist_retriever = False
        self.database_path = os.path.join(self.root, f"vector_store")
        self.cache_root = os.path.join(self.root, f"cache")
        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.cache_root, exist_ok=True)

        # 给定文档，然后切分成句子。句子再生成fragments  文档-句子=1:m， 句子-fragments=1:1
        self.doc_store = ChunkStore(self.embedding_model, self.database_path, mode='document')
        self.sentence_store = ChunkStore(self.embedding_model, self.database_path, mode='sentence')
        self.phrase_store = ChunkStore(self.embedding_model, self.database_path, mode='phrase')


        # self.key2content = {}

    def format_queries(self, batch_queries, search_mode='query_to_doc'):
        if self.model_mode == 'normal':
            return batch_queries
        elif self.model_mode == 'instruction':
            results = []
            for query in batch_queries:
                ins = get_query_instruction(search_mode)
                query = f"Instruct: {ins}\nQuery: {query}"
                results.append(query)
            return results
        else:
            raise ValueError("error model_mode normal/instruction")

    def embed_queries(self, batch_queries, search_mode='query_to_doc'):
        results = []
        if self.model_mode == 'normal':
            for query in batch_queries:
                results.append(self.embedding_model.embed_query(query))
        elif self.model_mode == 'instruction':
            for query in batch_queries:
                results.append(self.embedding_model.embed_query(query, search_mode))
        else:
            raise ValueError("error model_mode normal/instruction")

        return results

    def search(self, vector_store, query, allowed_ids: List[str] = None, top_k=5, score_threshold: float = 0.5):
        # {id, content, title, score}
        sequence_text, sequence_key, sequence_score = vector_store.search(query, allowed_ids=allowed_ids, top_k=top_k, score_threshold=score_threshold)
        return [sequence_text, sequence_key, sequence_score]

    def search_by_vector(self, vector_store, query_embedding, allowed_ids: List[str] = None, top_k=5, score_threshold: float = None):
        # {id, content, title, score}
        sequence_text, sequence_key, sequence_score = vector_store.search_by_vector(query_embedding, allowed_ids=allowed_ids, top_k=top_k, score_threshold=score_threshold)
        # 缓存结果
        return [sequence_text, sequence_key, sequence_score]

    def batch_search(self, batch_queries: List[str], batch_allowed_ids: List[List[str]] = None, top_k=5, score_threshold: float = None, search_mode='query_to_doc'):
        local_params = {'top_k': top_k, 'score_threshold': score_threshold}
        if batch_allowed_ids is None:
            batch_allowed_ids = [None]*len(batch_queries)

        # 首先对query进行编码
        idx2hit_key = self.check_search_cache(batch_queries, batch_allowed_ids, local_params, search_mode)

        # 存放所有消息的key
        batch_query_keys = []
        # 第二步，将待处理的消息进入队列进行处理
        sequence_query_key = []
        sequence_query = []
        sequence_allowed_ids = []
        for idx in range(len(batch_queries)):
            hit, key = idx2hit_key[idx]
            batch_query_keys.append(key)
            if not hit:
                sequence_query_key.append(key)
                sequence_query.append(batch_queries[idx])
                sequence_allowed_ids.append(batch_allowed_ids[idx])
                # sequence_query_embeddings.append(self.embedding_model.embed_query(batch_queries[idx]))

        if search_mode == 'query_to_doc' or search_mode == 'agument_query_to_doc':
            vector_store = self.doc_store
        elif search_mode == 'query_to_sentence':
            vector_store = self.sentence_store
        elif search_mode == 'phrase_to_phrase':
            vector_store = self.phrase_store
        else:
            raise ValueError("Search mode not supported")

        if len(sequence_query_key):
            print("查看检索缓存")
            sequence_formated_queries = self.embed_queries(sequence_query, search_mode)
            # 开始搜索
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.search_by_vector, vector_store, query, allowed_ids, top_k, score_threshold) for query, allowed_ids in tqdm(zip(sequence_formated_queries, sequence_allowed_ids), total=len(sequence_query), desc=f'Search {search_mode}', disable=not self.enable_search_tqdm)]  # 提交任务
                results = [future.result() for future in futures]  # 获取所有任务的结果
            # results = []
            # for query, allowed_ids in tqdm(zip(sequence_formated_queries, sequence_allowed_ids),
            #                                total=len(sequence_query), desc=f'Search {search_mode}',
            #                                disable=not self.enable_search_tqdm):
            #     results.append(self.search(vector_store, query, allowed_ids, top_k, score_threshold))

            sequence_documents = []     # List[List[Dict]]
            for sequence_text, sequence_key, sequence_score in results:
                # for text, key in zip(sequence_text, sequence_key):
                #     if key not in self.key2content:
                #         self.key2content[key] = text
                sequence_documents.append((sequence_key, sequence_score))

            # 即存在没有缓存的数据， 构建缓存数据
            cache_sequence_data = []
            for query_key, sorted_documents in zip(sequence_query_key, sequence_documents):
                cache_sequence_data.append({'key': query_key, 'sorted_documents': sorted_documents})

            # save_cache(self.cache_root, cache_sequence_data, f'search_{search_mode}')
            save_cache(cache_root=self.cache_root,
                       sequences=cache_sequence_data,
                       filename=f'search_{search_mode}',
                       table_name='cache')

        batch_sorted_documents_metadata = load_cache(cache_root=self.cache_root,
                                                     sequence_key=batch_query_keys,
                                                     filename=f'search_{search_mode}',
                                                     table_name='cache')


        batch_sorted_documents = []
        for metadata in batch_sorted_documents_metadata:
            sequence = []
            sequence_key, sequence_score = metadata['sorted_documents']
            for key, score in zip(sequence_key, sequence_score):
                # print(list(vector_store.key2content.keys())[:2])
                # print(key)
                # print('......')
                text = vector_store.key2content.get(key, None)
                # print("??", text)
                if text is None:
                    continue
                sequence.append({'content': text, 'key': key, 'score': score})
            batch_sorted_documents.append(sequence)
        return batch_sorted_documents

    def add_documents(self, documents: List[Dict], chunk_type='doc'):
        '''将文档加入到向量库中'''
        if chunk_type == 'doc':
            vector_store = self.doc_store
            desc = "Updating Documents to Vector database"
        elif chunk_type == 'sentence':
            vector_store = self.sentence_store
            desc = "Updating Sentences to Vector database"
        elif chunk_type == 'phrase':
            vector_store = self.phrase_store
            desc = "Updating Phrases to Vector database"
        else:
            raise ValueError(f'Unknown chunk_type {chunk_type}')

        vector_store.add_documents(documents, desc=desc)
        vector_store.save_database()
    def add_documents_dict(self, documents: Dict, chunk_type='doc'):
        '''将文档加入到向量库中'''
        if chunk_type == 'doc':
            vector_store = self.doc_store
            desc = "Updating Documents to Vector database"
        elif chunk_type == 'sentence':
            vector_store = self.sentence_store
            desc = "Updating Sentences to Vector database"
        elif chunk_type == 'phrase':
            vector_store = self.phrase_store
            desc = "Updating Phrases to Vector database"
        else:
            raise ValueError(f'Unknown chunk_type {chunk_type}')

        vector_store.add_documents_dict(documents, desc=desc)
        vector_store.save_database()
    def encode_search_key(self, query, allowed_ids, local_params, search_mode='query_to_doc'):
        if allowed_ids is None:
            allowed_ids = []
        allowed_ids = sorted(allowed_ids)
        key_data = {
            'query': query,
            'allowed_ids': allowed_ids,
            'local': local_params
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = generate_key(key_str, prefix=f'search_{search_mode}-')
        return key_hash

    def check_search_cache(self, batch_queries, batch_allowed_ids, local_params, search_mode='query_to_doc'):
        all_search_keys = self.get_search_keys(search_mode)

        key_list = []
        key2idxs = defaultdict(set)  # key转原本序列号
        idx2hit_key = {}  # 序列号对应的cache状态
        for idx, (query, allowed_ids) in enumerate(zip(batch_queries, batch_allowed_ids)):
            key = self.encode_search_key(query, allowed_ids, local_params, search_mode)
            key_list.append(key)
            key2idxs[key].add(idx)
            idx2hit_key[idx] = (False, key)

        key_set = set(key2idxs.keys())

        keys_in_cache = all_search_keys & key_set
        for key in keys_in_cache:
            for idx in key2idxs[key]:
                idx2hit_key[idx] = (True, key)

        return idx2hit_key

    def get_search_keys(self, search_mode='query_to_doc'):
        if search_mode not in ['query_to_doc', 'query_to_sentence', 'phrase_to_phrase', 'agument_query_to_doc']:
            raise ValueError('Search_mode must be query_to_doc/query_to_sentence/phrase_to_phrase/agument_query_to_doc')

        cache_keys = set(get_cache_keys(self.cache_root, f'search_{search_mode}', 'cache'))
        return set(cache_keys)