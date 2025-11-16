'''向量数据库
功能简介：
1. 向量存储：给定多个文档，将其编译成向量，进行存储(请看注意1）
2. 检索：根据一个问题，有两种检索： <1>从全局文件中检索, <2>给定文件范围检索（即给定范围内所有文档的Keys）
注意：
1. 向量保存问题： 为了避免多次调用add_documents函数，导致频繁保存文件。需要自己手动调用save_database在合适的时间保存，不会自动保存。

'''

from langchain_chroma import Chroma
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
        self.database_path = database_path #os.path.join(self.root, f'vector_store_{embedding_model_name}')

        self.vector_store = Chroma(
            collection_name=self.mode,
            embedding_function=self.embedding_model,
            persist_directory=self.database_path,  # Where to save data locally, remove if not necessary
        )
        # 获取所有的chunk_key
        self.chunk_keys = self.get_chunk_keys()

    def get_chunk_keys(self):
        # 获取所有的chunk_key
        metadata = self.vector_store.get()['metadatas']
        all_keys = set()
        for item in metadata:
            all_keys.add(item['key'])

        return all_keys

    # 增加数据
    def add_documents(self, documents: List[Dict], desc='向量库更新', batch_size=5000):
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

            self.chunk_keys.add(key)

        for i in tqdm(range(0, len(unrecorded_documents), batch_size), desc=desc, disable=not self.enable_add_tqdm):
            batch = unrecorded_documents[i:i + batch_size]
            self.vector_store.add_documents(documents=batch)

    def add_documents_dict(self, documents: Dict, desc='向量库更新', batch_size=5000):
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

            self.chunk_keys.add(key)

        for i in tqdm(range(0, len(unrecorded_documents), batch_size), desc=desc, disable=not self.enable_add_tqdm):
            batch = unrecorded_documents[i:i + batch_size]
            self.vector_store.add_documents(documents=batch)

    def strip_search_res(self, document, score):
        item = {'content': document.page_content, 'score': float(score)}
        item.update(document.metadata)
        return item

    def search(self, query: str, allowed_ids: List[str] = None, top_k=5, score_threshold: float = 0.5):
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
        relevance_score_fn = self.vector_store._select_relevance_score_fn()
        docs_and_scores = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

        # 格式化结果
        filter_results = []
        keys = []
        for document, score in docs_and_scores:
            if score < score_threshold:
                # 如果score小于score_threshold，后面数据的也不用看了，都小于
                break
            item = self.strip_search_res(document, score)
            filter_results.append(item)
            keys.append(item['key'])

        return filter_results, keys

    def search_by_vector(self, query_vector, allowed_ids: List[str] = None, top_k=5, score_threshold: float = None):
        if allowed_ids is None:
            # 即检索范围为：库中所有文档
            if isinstance(top_k, float):
                top_k = int(len(self.chunk_keys) * top_k)
            docs_and_scores = self.vector_store.similarity_search_by_vector_with_relevance_scores(query_vector, k=top_k)
        else:
            # 限制检索范围: allowed_ids
            if isinstance(top_k, float):
                top_k = int(len(allowed_ids) * top_k)
            filter_condition = {'key': {'$in': allowed_ids}} if allowed_ids else None
            docs_and_scores = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                query_vector,
                k=top_k,
                filter=filter_condition,  # lambda metadata: metadata.get('key', 'None') in allowed_ids,
            )
        relevance_score_fn = self.vector_store._select_relevance_score_fn()
        docs_and_scores = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

        # 格式化结果
        filter_results = []
        keys = []

        for document, score in docs_and_scores:
            if score_threshold is not None and score < score_threshold:
                # 如果score小于score_threshold，后面数据的也不用看了，都小于
                break
            item = self.strip_search_res(document, score)
            filter_results.append(item)
            keys.append(item['key'])

        return filter_results, keys
