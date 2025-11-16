'''向量数据库
功能简介：
1. 向量存储：给定多个文档，将其编译成向量，进行存储(请看注意1）
2. 检索：根据一个问题，有两种检索： <1>从全局文件中检索, <2>给定文件范围检索（即给定范围内所有文档的Keys）
注意：
1. 向量保存问题： 为了避免多次调用add_documents函数，导致频繁保存文件。需要自己手动调用save_database在合适的时间保存，不会自动保存。

'''

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from utils.file_manager import safe_save_file
from langchain.schema import Document
from typing import List, Dict, Optional
import os
from tqdm import tqdm
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

class ChunkStore:
    def __init__(self,
                 embedding_model,
                 database_path: str,
                 mode='document',
                 enable_add_tqdm=True):
        self.embedding_model = embedding_model
        # self.root = database_root  # 数据库根目录
        self.enable_add_tqdm = enable_add_tqdm      # 是否开启 添加文档的tqdm
        self.mode = mode
        self.database_path = os.path.join(database_path, f'{mode}')

        self.need_update_store = False

        self.vector_store = self.load_database()

        # 获取所有的chunk_key
        self.chunk_keys, self.key2content = self.get_chunk_keys()

    def get_chunk_keys(self):
        if self.vector_store is None:
            return set(), {}

        all_documents = list(self.vector_store.docstore._dict.values())
        all_keys = set()
        all_key2content = {}
        for doc in all_documents:
            key = doc.metadata['key']
            content = doc.page_content
            all_keys.add(key)
            all_key2content[key] = content

        return all_keys, all_key2content

    def save_database(self):
        if not self.need_update_store or self.vector_store is None:
            return

        def save_func(filename):
            self.vector_store.save_local(filename)

        save_status = safe_save_file(self.database_path, save_func)
        self.need_update_store = not save_status

    def load_database(self):
        try:
            # 加载索引
            return FAISS.load_local(self.database_path, self.embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            return None

    # 增加数据
    def add_documents(self, documents: List[Dict], desc='向量库更新', batch_size=10000):
        # documents : {key, content, 其他key}
        unrecorded_documents = []  # 没有存储的文档
        for dict_item in documents:
            key = dict_item['key']
            if key in self.chunk_keys:
                continue
            content = dict_item['content']
            metadata = {}
            for k, v in dict_item.items():
                if k in ['content']:
                    continue
                metadata[k] = v

            unrecorded_documents.append(Document(page_content=content, metadata=metadata))
            self.key2content[key] = content
            self.chunk_keys.add(key)

        if len(unrecorded_documents):
            self.need_update_store = True

        for i in tqdm(range(0, len(unrecorded_documents), batch_size), desc=desc, disable=not self.enable_add_tqdm):
            batch = unrecorded_documents[i:i + batch_size]
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents=batch,
                                                        embedding=self.embedding_model,
                                                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            else:
                self.vector_store.add_documents(batch)

    def add_documents_dict(self, documents: Dict, desc='向量库更新', batch_size=10000):
        # documents : {key:[], , content:[], 其他key:[]}
        unrecorded_documents = []  # 没有存储的文档
        metdata_columns = set(documents.keys()) - {'content'}
        for idx, key in enumerate(documents['key']):
            if key in self.chunk_keys:
                continue
            content = documents['content'][idx]

            metadata = {}
            for k in metdata_columns:
                metadata[k] = documents[k][idx]
            unrecorded_documents.append(Document(page_content=content, metadata=metadata))
            self.key2content[key] = content
            self.chunk_keys.add(key)

        if len(unrecorded_documents):
            self.need_update_store = True
        # 分批添加
        for i in tqdm(range(0, len(unrecorded_documents), batch_size), desc=desc, disable=not self.enable_add_tqdm):
            batch = unrecorded_documents[i:i + batch_size]
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents=batch,
                                                         embedding=self.embedding_model,
                                                         distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            else:
                self.vector_store.add_documents(batch)

    def strip_search_res(self, document, score):
        item = {'content': document.page_content, 'score': float(score)}
        item.update(document.metadata)

        return document.page_content, document.metadata['key'], float(score)
        # return item

    def search(self, query: str, allowed_ids: List[str] = None, top_k=5, score_threshold: float = None):
        if allowed_ids is None:
            # 即检索范围为：库中所有文档
            if isinstance(top_k, float):
                top_k = int(len(self.chunk_keys) * top_k)
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        else:
            # 限制检索范围: allowed_ids
            if isinstance(top_k, float):
                top_k = int(len(allowed_ids) * top_k)
            filter_condition = {'key': {'$in': allowed_ids}} if allowed_ids else None
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_condition,  # lambda metadata: metadata.get('key', 'None') in allowed_ids,
            )
        # relevance_score_fn = self.vector_store._select_relevance_score_fn()
        # docs_and_scores = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

        # 格式化结果
        sequence_text = []
        sequence_key = []
        sequence_score = []

        for document, score in docs_and_scores:
            if score_threshold is not None and score < score_threshold:
                # 如果score小于score_threshold，后面数据的也不用看了，都小于
                break
            content, key, score = self.strip_search_res(document, score)
            sequence_text.append(content)
            sequence_key.append(key)
            sequence_score.append(score)
            # filter_results.append(item)
            # keys.append(item['key'])
        return sequence_text, sequence_key, sequence_score
        # return filter_results, keys

    def search_by_vector(self, query_vector, allowed_ids: List[str] = None, top_k=5, score_threshold: float = None):
        if allowed_ids is None:
            # 即检索范围为：库中所有文档
            if isinstance(top_k, float):
                top_k = int(len(self.chunk_keys) * top_k)
            docs_and_scores = self.vector_store.similarity_search_with_score_by_vector(query_vector, k=top_k)
        else:
            # 限制检索范围: allowed_ids
            if isinstance(top_k, float):
                top_k = int(len(allowed_ids) * top_k)
            filter_condition = {'key': {'$in': allowed_ids}} if allowed_ids else None
            docs_and_scores = self.vector_store.similarity_search_with_score_by_vector(
                query_vector,
                k=top_k,
                filter=filter_condition,  # lambda metadata: metadata.get('key', 'None') in allowed_ids,
            )
        # relevance_score_fn = self.vector_store._select_relevance_score_fn()
        # docs_and_scores = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

        # 格式化结果
        # filter_results = []
        # keys = []
        sequence_text = []
        sequence_key = []
        sequence_score = []

        for document, score in docs_and_scores:
            if score_threshold is not None and score < score_threshold:
                # 如果score小于score_threshold，后面数据的也不用看了，都小于
                break
            content, key, score = self.strip_search_res(document, score)
            sequence_text.append(content)
            sequence_key.append(key)
            sequence_score.append(score)
            # filter_results.append(item)
            # keys.append(item['key'])
        return sequence_text, sequence_key, sequence_score
        # return filter_results, keys
