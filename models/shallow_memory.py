'''构建伪知识图谱'''
from collections import defaultdict
from igraph import Graph
from typing import List
from config import HotpotQAConfig as default_config
from utils.text_manager import check_numbers_relations, has_uppercase, has_number, strip_query, \
    batch_get_similar_keywords
from utils.file_manager import save_pickle, load_pickle
from utils.log_manager import get_logger
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import shutil
logger = get_logger(__name__)


class ShallowMemory:
    def __init__(self, extractor, base_retriever, filter, query_optimizer,template_manager, config=default_config, remove_data=False):
        self.extractor = extractor
        self.retriever = base_retriever
        self.filter = filter
        self.query_optimizer = query_optimizer
        self.template_manager = template_manager
        self.config = config
        print("???????", self.config.dataset.dataset_name, self.config.dataset.output_root)

        self.root = os.path.join(self.config.dataset.output_root, 'shallow_memory')
        if os.path.exists(self.root) and remove_data:
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)


        self.max_workers = self.config.rag.max_workers

        self.load_memory()

    # 从corpus中抽取实体构建浅层记忆图
    def corpus_to_memory(self):
        corpus_path = self.config.dataset.corpus_path
        logger.info("NER Corpus...")
        corpus_doc_info, corpus_sentence_info, corpus_phrase_info = self.extractor.ner_corpus_to_parquet(corpus_path)
        logger.info(
            f"Corpus Doc {len(corpus_doc_info['key'])}, Sentecne {len(corpus_sentence_info['key'])}, Phrase {len(corpus_phrase_info['key'])}")

        logger.info("Preprocessing data for Corpus-to-Shallow Memory conversion...")

        all_docs = {}
        all_sentences = {}
        all_phrases = {}
        mapping_key_sentence2phrases = {}
        mapping_key_doc2sentences = {}

        for doc_key, doc_text, sentence_keys in zip(corpus_doc_info['key'], corpus_doc_info['content'],
                                                    corpus_doc_info['sentence_keys']):
            all_docs[doc_key] = doc_text
            mapping_key_doc2sentences[doc_key] = sentence_keys

        for sentence_key, sentence_text, phrase_keys in zip(corpus_sentence_info['key'],
                                                            corpus_sentence_info['content'],
                                                            corpus_sentence_info['phrase_keys']):
            all_sentences[sentence_key] = sentence_text
            mapping_key_sentence2phrases[sentence_key] = set(phrase_keys)

        for phrase_key, phrase_text in zip(corpus_phrase_info['key'], corpus_phrase_info['content']):
            all_phrases[phrase_key] = phrase_text

        # 首先构建图
        key2node = {}
        edges = set()
        graph_keys = []
        graph_texts = []
        set_doc_nodes = set()
        set_sentence_nodes = set()
        set_phrase_nodes = set()
        node_doc2sentences = {}
        node_sentence2docs = {}
        node_sentence2phrases = {}

        # 开始构建文档节点信息
        for doc_key, doc_text in all_docs.items():
            node_id = len(key2node)
            key2node[doc_key] = node_id
            graph_keys.append(doc_key)
            graph_texts.append(doc_text)
            set_doc_nodes.add(node_id)

        # 开始构建节点信息
        for sent_key, sent_text in all_sentences.items():
            node_id = len(graph_keys)
            key2node[sent_key] = node_id
            graph_keys.append(sent_key)
            graph_texts.append(sent_text)
            set_sentence_nodes.add(node_id)

        for phrase_key, phrase_text in all_phrases.items():
            node_id = len(graph_keys)
            key2node[phrase_key] = node_id
            graph_keys.append(phrase_key)
            graph_texts.append(phrase_text)
            set_phrase_nodes.add(node_id)

        # 开始构建边
        for doc_key in all_docs:
            relevant_sentence_keys = mapping_key_doc2sentences[doc_key]
            doc_node = key2node[doc_key]
            sentence_nodes = []
            edges_doc2sentence = []
            for idx, sent_key in enumerate(relevant_sentence_keys):
                sentence_node = key2node[sent_key]
                sentence_nodes.append(sentence_node)
                edges_doc2sentence.append((doc_node, sentence_node))
                if sent_key not in node_sentence2docs:
                    node_sentence2docs[sentence_node] = set()
                node_sentence2docs[sentence_node].add(doc_node)
            node_doc2sentences[doc_node] = sentence_nodes

            edges.update(edges_doc2sentence)

        for sent_key, relevant_phrase_keys in mapping_key_sentence2phrases.items():
            sent_node = key2node[sent_key]
            # 这里顺序无关
            phrase_nodes = []
            sent_edges = []
            for idx, phrase_key in enumerate(relevant_phrase_keys):
                phrase_node = key2node[phrase_key]
                phrase_nodes.append(phrase_node)
                sent_edges.append((sent_node, phrase_node, idx))

            node_sentence2phrases[sent_node] = set(phrase_nodes)
            sent_edges = [(sent_node, key2node[phrase_key]) for phrase_key in relevant_phrase_keys]
            edges.update(sent_edges)

        # 创建图并设置属性
        self.graph = Graph(directed=False)
        self.graph.add_vertices(len(graph_keys))
        self.graph.vs['id'] = list(range(len(graph_keys)))
        self.graph.vs['key'] = graph_keys
        self.graph.vs['content'] = graph_texts
        edges = list(edges)
        self.graph.add_edges(edges)

        self.np_graph_keys = np.array(graph_keys)
        self.np_graph_texts = np.array(graph_texts)
        self.set_graph_edges = set(self.graph.get_edgelist())

        print(f"Number of nodes: {self.graph.vcount()}")
        print(f"Number of edges: {self.graph.ecount()}")

        logger.info("Corpus Data To Embedding Database ...")

        del corpus_doc_info['sentence_keys']
        del corpus_doc_info['phrase_keys']
        del corpus_sentence_info['phrase_keys']
        self.retriever.add_documents_dict(corpus_doc_info, chunk_type='doc')
        self.retriever.add_documents_dict(corpus_sentence_info, chunk_type='sentence')
        self.retriever.add_documents_dict(corpus_phrase_info, chunk_type='phrase')

        self.key2node = key2node
        self.set_doc_nodes = set_doc_nodes
        self.set_sentence_nodes = set_sentence_nodes
        self.set_phrase_nodes = set_phrase_nodes
        self.node_doc2sentences = node_doc2sentences
        self.node_sentence2docs = node_sentence2docs
        self.node_sentence2phrases = node_sentence2phrases

        # 保存图数据
        self.save_memory()

        return self.graph

    def texts_to_memory(self, texts):
        # documents =[{'content': text, 'key': key, 'score': score}...]
        logger.info("NER Texts ...")
        allowed_texts = []
        for text in texts:
            if text not in self.np_graph_texts:
                allowed_texts.append(text)
        logger.info(f"update {len(allowed_texts)} docs to Memory")
        corpus_doc_info, corpus_sentence_info, corpus_phrase_info = self.extractor.ner_documents_to_sqlite(allowed_texts)
        logger.info(f"Texts to Memory:  Doc {len(corpus_doc_info['key'])}, Sentecne {len(corpus_sentence_info['key'])}, Phrase {len(corpus_phrase_info['key'])}")
        logger.info("Preprocessing data for Text-to-Shallow Memory conversion...")

        all_docs = {}
        all_sentences = {}
        all_phrases = {}
        mapping_key_sentence2phrases = {}
        mapping_key_doc2sentences = {}

        for doc_key, doc_text, sentence_keys in zip(corpus_doc_info['key'], corpus_doc_info['content'],
                                                    corpus_doc_info['sentence_keys']):
            all_docs[doc_key] = doc_text
            mapping_key_doc2sentences[doc_key] = sentence_keys

        for sentence_key, sentence_text, phrase_keys in zip(corpus_sentence_info['key'],
                                                            corpus_sentence_info['content'],
                                                            corpus_sentence_info['phrase_keys']):
            all_sentences[sentence_key] = sentence_text
            mapping_key_sentence2phrases[sentence_key] = set(phrase_keys)

        for phrase_key, phrase_text in zip(corpus_phrase_info['key'], corpus_phrase_info['content']):
            all_phrases[phrase_key] = phrase_text

        # self.graph.vs['id'] = list(range(len(graph_keys)))
        # self.graph.vs['key'] = graph_keys
        # self.graph.vs['content'] = graph_texts
        new_ids = []
        new_keys = []
        new_contents = []
        new_edges = set()
        # 首先构建图
        for doc_key, doc_text in all_docs.items():
            if doc_key in self.key2node:
                continue
            node_id = len(self.key2node)
            self.key2node[doc_key] = node_id
            self.set_doc_nodes.add(node_id)
            new_ids.append(node_id)
            new_keys.append(doc_key)
            new_contents.append(doc_text)

        # 开始构建节点信息
        for sent_key, sent_text in all_sentences.items():
            if sent_key in self.key2node:
                continue
            node_id = len(self.key2node)
            self.key2node[sent_key] = node_id
            new_ids.append(node_id)
            new_keys.append(sent_key)
            new_contents.append(sent_text)

        for phrase_key, phrase_text in all_phrases.items():
            if phrase_key in self.key2node:
                continue
            node_id = len(self.key2node)
            self.key2node[phrase_key] = node_id
            new_ids.append(node_id)
            new_keys.append(phrase_key)
            new_contents.append(phrase_text)

        # 开始构建边
        for doc_key in all_docs:
            relevant_sentence_keys = mapping_key_doc2sentences[doc_key]
            doc_node = self.key2node[doc_key]
            sentence_nodes = []
            edges_doc2sentence = []
            for idx, sent_key in enumerate(relevant_sentence_keys):
                sentence_node = self.key2node[sent_key]
                sentence_nodes.append(sentence_node)
                edges_doc2sentence.append((doc_node, sentence_node))
                if sent_key not in self.node_sentence2docs:
                    self.node_sentence2docs[sentence_node] = set()
                self.node_sentence2docs[sentence_node].add(doc_node)
            self.node_doc2sentences[doc_node] = sentence_nodes

            new_edges.update(edges_doc2sentence)

        for sent_key, relevant_phrase_keys in mapping_key_sentence2phrases.items():
            sent_node = self.key2node[sent_key]
            # 这里顺序无关
            phrase_nodes = []
            sent_edges = []
            for idx, phrase_key in enumerate(relevant_phrase_keys):
                phrase_node = self.key2node[phrase_key]
                phrase_nodes.append(phrase_node)
                sent_edges.append((sent_node, phrase_node, idx))

            self.node_sentence2phrases[sent_node] = set(phrase_nodes)
            sent_edges = [(sent_node, self.key2node[phrase_key]) for phrase_key in relevant_phrase_keys]
            new_edges.update(sent_edges)

        # 创建图并设置属性
        old_edges = self.graph.ecount()
        self.graph.add_vertices(len(new_ids), attributes={'id': new_ids, 'key': new_keys, 'content': new_contents})
        edges = list(new_edges)
        self.graph.add_edges(edges)
        self.np_graph_keys = np.array(self.graph.vs['key'])
        self.np_graph_texts = np.array(self.graph.vs['content'])
        self.set_graph_edges = set(self.graph.get_edgelist())

        print(f"New Number of nodes: {len(new_ids)}")
        print(f"New Number of edges: {self.graph.ecount() - old_edges}")

        logger.info("Corpus Data To Embedding Database ...")

        del corpus_doc_info['sentence_keys']
        del corpus_doc_info['phrase_keys']
        del corpus_sentence_info['phrase_keys']
        self.retriever.add_documents_dict(corpus_doc_info, chunk_type='doc')
        # self.retriever.add_documents_dict(corpus_sentence_info, chunk_type='sentence')
        self.retriever.add_documents_dict(corpus_phrase_info, chunk_type='phrase')

        if len(new_ids):
            self.need_save = True

        # 邻居节点
        self.all_degrees = np.array(self.graph.degree())
        # 生成ppr相关的子图
        self.ppr_graph, self.local2global_nodes = self.get_ppr_graph()
        self.node_idfs = 1 + 1 / (self.all_degrees + 1)  # np.log(len(self.set_sentence_nodes)
    def save_memory(self):
        # 然后保存igraph和节点数据
        logger.info("Save Shallow Memory to local pickle ...")
        save_data = {'set_doc_nodes': self.set_doc_nodes,
                     'set_sentence_nodes': self.set_sentence_nodes,
                     'set_phrase_nodes': self.set_phrase_nodes,
                     'node_doc2sentences': self.node_doc2sentences,
                     'node_sentence2phrases': self.node_sentence2phrases,
                     'node_sentence2docs': self.node_sentence2docs}

        save_pickle(save_data, os.path.join(self.root, 'info.pkl'))
        self.graph.write_pickle(os.path.join(self.root, "struct.pkl"))
        logger.info("Save Shallow Memory OK !")

    def load_memory(self):
        try:
            logger.info("Load Shallow Memory ...")
            graph_info = load_pickle(os.path.join(self.root, 'info.pkl'))
            self.graph = Graph.Read_Pickle(os.path.join(self.root, "struct.pkl"))
            # 开始构建图索引
            self.set_doc_nodes = graph_info['set_doc_nodes']
            self.set_sentence_nodes = graph_info['set_sentence_nodes']
            self.set_phrase_nodes = graph_info['set_phrase_nodes']
            self.node_doc2sentences = graph_info['node_doc2sentences']
            self.node_sentence2phrases = graph_info['node_sentence2phrases']
            self.node_sentence2docs = graph_info['node_sentence2docs']
            self.np_graph_keys = np.array(self.graph.vs['key'])
            self.np_graph_texts = np.array(self.graph.vs['content'])
            self.set_graph_edges = set(self.graph.get_edgelist())
            self.key2node = {key: idx for idx, key in enumerate(self.np_graph_keys)}
            print(f"Number of nodes of shallow memory: {self.graph.vcount()}")
            print(f"Number of edges of shallow memory: {self.graph.ecount()}")
            logger.info("Load Shallow Memory OK!")
        except:
            self.corpus_to_memory()
            logger.info("Corpus to Shallow Memory OK!")

        # 要在哪些类型节点运行ppr
        self.ppr_structure = self.config.rag.ppr_structure
        # 邻居节点
        self.all_degrees = np.array(self.graph.degree())
        # 生成ppr相关的子图
        self.ppr_graph, self.local2global_nodes = self.get_ppr_graph()
        self.node_idfs =  1 + 1 / (self.all_degrees + 1) # np.log(len(self.set_sentence_nodes)

    def get_ppr_graph(self):
        # 构建子图
        if self.ppr_structure == 's2e':
            target_nodes = self.set_sentence_nodes | self.set_phrase_nodes
        elif self.ppr_structure == 'd2s2e':
            target_nodes = self.set_doc_nodes | self.set_sentence_nodes | self.set_phrase_nodes
        else:
            raise ValueError(f"Unknown type: {self.ppr_structure} [s2e / d2s2e]")

        subgraph = self.graph.induced_subgraph(list(target_nodes))
        local2global_nodes = [_id for _id in subgraph.vs['id']]

        return subgraph, local2global_nodes

    def update_shallow_memory_mapping(self):
        # 如果图有变动，跟igraph图相关的参数需要重新获取
        self.np_graph_keys = np.array(self.graph.vs['key'])
        self.np_graph_texts = np.array(self.graph.vs['content'])
        self.set_graph_edges = set(self.graph.get_edgelist())
        self.all_degrees = np.array(self.graph.degree())
        self.node_idfs = 1 + 1 / (self.all_degrees + 1)

    def update_shallow_memory_with_phrases(self, phrases_info_dict):
        # 新增phrase，导致memory更新
        # {phrase_text: set(sentence_nodes)}  phrase - sentence
        old_edge_num = self.graph.ecount()
        old_node_num = self.graph.vcount()

        # 新增节点
        new_phrase_texts = set()
        for phrase in phrases_info_dict:
            phrase_key = self.extractor.get_phrase_key(phrase)
            if phrase_key not in self.np_graph_keys:
                new_phrase_texts.add(phrase)
        new_phrase_texts = list(new_phrase_texts)
        new_phrase_keys = [self.extractor.get_phrase_key(text) for text in new_phrase_texts]

        self.graph.add_vertices(len(new_phrase_keys))
        new_ids = list(range(old_node_num, old_node_num + len(new_phrase_keys)))
        for key, node_id in zip(new_phrase_keys, new_ids):
            self.key2node[key] = node_id
            self.set_phrase_nodes.add(node_id)

        self.graph.vs[old_node_num:]['id'] = new_ids
        self.graph.vs[old_node_num:]['key'] = new_phrase_keys
        self.graph.vs[old_node_num:]['content'] = new_phrase_texts

        edges = set()
        # 开始增加新边
        for phrase, set_related_sentence_nodes in phrases_info_dict.items():
            phrase_key = self.extractor.get_phrase_key(phrase)
            phrase_node = self.key2node[phrase_key]
            sent_edges = []
            for sent_node in set_related_sentence_nodes:
                if sent_node not in self.node_sentence2phrases:
                    self.node_sentence2phrases[sent_node] = set()
                self.node_sentence2phrases[sent_node].add(phrase_node)
                sent_edges.append((sent_node, phrase_node))
            edges.update(sent_edges)

        edges = list(edges)
        self.graph.add_edges(edges)
        add_node_num = self.graph.vcount() - old_node_num
        add_edge_num = self.graph.ecount() - old_edge_num
        if add_node_num or add_edge_num:
            logger.info(f"当前新增 {add_node_num}个实体节点, {add_edge_num}条边")

        self.update_shallow_memory_mapping()
        # 一旦图有更新，重新获得子图
        self.ppr_graph, self.local2global_nodes = self.get_ppr_graph()


        # 更新向量库
        docs = {'key':[], 'content':[]}
        for key, text in zip(new_phrase_keys, new_phrase_texts):
            docs['key'].append(key)
            docs['content'].append(text)
        self.retriever.add_documents_dict(docs, chunk_type='phrase')

        # 保存图
        # self.save_memory()

    def batch_init_graph_scores(self, batch_query_solutions):
        # for query_solution in batch_query_solutions:
        #     self.init_graph_scores(query_solution)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.init_graph_scores, query_solution) for query_solution in
                       batch_query_solutions]
            batch_query_solutions = [future.result() for future in futures]
        return batch_query_solutions

    def filter_knowledges(self, batch_query_solutions):
        # 筛选前，需要加载带筛选的文档
        for solution in batch_query_solutions:
            fact_texts = []  # List[List]
            fact_nodes = []  # List[List]
            if solution.processed_document_nodes is None:
                solution.processed_document_nodes = set()

            for doc, doc_key in zip(solution.ranked_docs, solution.ranked_doc_keys):
                # 被处理过的doc_key就不用再筛选了
                # if doc_key in solution.processed_document_keys:
                #     continue
                if len(fact_texts) >= self.config.rag.filter_doc_num:
                    # 已经拿到了要筛选的数量，终止
                    break

                doc_node = self.key2node[doc_key]
                sentence_nodes = self.node_doc2sentences[doc_node]
                sentence_texts = self.np_graph_texts[sentence_nodes]
                fact_texts.append(sentence_texts)
                fact_nodes.append(sentence_nodes)
                solution.processed_document_nodes.add(doc_node)
            # 加载
            solution.unprocessed_fact_nodes = fact_nodes
            solution.unprocessed_fact_texts = fact_texts
            solution.evidence_update_status = False

        # 开始filter
        self.filter.batch_filter(batch_query_solutions)

    def match_relevant_phrase_nodes(self, match_phrases: List[str]):
        # Step 1: 批量模糊搜索，获取每个实体的候选匹配
        search_results = self.retriever.batch_search(
            match_phrases,
            top_k=self.config.shallow_memory.match_node_num,
            score_threshold=self.config.shallow_memory.match_threshold,
            search_mode='phrase_to_phrase'
        )

        # Step 2: 构建等价实体映射 phrase -> {llm_phrase}
        same_graph_phrases_to_match_phrases = defaultdict(set)
        other_graph_phrases = set()
        graph_phrase2node = {}
        sameAs_score = self.config.shallow_memory.match_same_threshold

        # 如果相似度>=sameAs_score,认为从图中找到了等价的phrase，如果小于，可能存在相似的phrase
        for match_phrase, results in zip(match_phrases, search_results):
            same_graph_phrases_to_match_phrases[match_phrase].add(match_phrase)
            for res in results:
                phrase = res['content']
                score = res['score']
                key = res['key']
                node = self.key2node[key]
                graph_phrase2node[phrase] = node
                # 是否存在数值不一致情况，因为语义相似度匹配可能导致1981和1980存在高相似
                if not check_numbers_relations(match_phrase, phrase):
                    continue
                if score >= sameAs_score:
                    same_graph_phrases_to_match_phrases[phrase].add(match_phrase)
                else:
                    other_graph_phrases.add(phrase)

        # Step 4: 批量获取相似实体
        all_same_graph_phrases = list(same_graph_phrases_to_match_phrases.keys())
        # 根据模糊匹配找到相似的phrase
        similar_phrases_dict = batch_get_similar_keywords(all_same_graph_phrases, list(other_graph_phrases))
        # Step 5: 构建最终映射
        match_phrase_to_similar_phrase_nodes = defaultdict(set)
        match_phrase_to_same_phrase_nodes = defaultdict(set)
        for search_phrase in all_same_graph_phrases:
            match_phrases = same_graph_phrases_to_match_phrases[search_phrase]
            key = self.extractor.get_phrase_key(search_phrase)
            search_node = self.key2node.get(key, None)

            # 获取相似实体并转换为节点编号
            similar_phrases = similar_phrases_dict.get(search_phrase, [])
            similar_nodes = set()
            for phrase in similar_phrases:
                node = graph_phrase2node[phrase]
                similar_nodes.add(node)
            # similar_nodes = np.where(np.isin(self.graph_texts, list(similar_phrases)))[0]
            # 更新映射
            for math_phrase in match_phrases:
                match_phrase_to_similar_phrase_nodes[math_phrase].update(similar_nodes)
                if search_node is not None:
                    match_phrase_to_same_phrase_nodes[math_phrase].add(search_node)

        # Step 6: 划分实体词 vs 概念词
        entity_match_phrases = set()
        concept_match_phrases = set()
        unprocessed_phrases = set()  # 单一词汇需要额外处理，可能是概念词Film大写了
        for phrase in match_phrase_to_similar_phrase_nodes:
            if has_uppercase(phrase) or has_number(phrase):
                entity_match_phrases.add(phrase)
            else:
                concept_match_phrases.add(phrase)
            if len(phrase) == 1:
                unprocessed_phrases.add(phrase)
        similar_phrase_res = batch_get_similar_keywords(list(concept_match_phrases), list(unprocessed_phrases))
        belong_to_concept_match_phrases = set()
        for search_phrase, similar_phrases in similar_phrase_res.items():
            belong_to_concept_match_phrases.update(similar_phrases)
        concept_match_phrases |= belong_to_concept_match_phrases
        entity_match_phrases -= concept_match_phrases

        return (
            entity_match_phrases,
            concept_match_phrases,
            match_phrase_to_same_phrase_nodes,
            match_phrase_to_similar_phrase_nodes
        )

    def classify_phrase_nodes(self, batch_query_solutions):
        # 对phrase进行分类，按照evidence级别，phrase(entity/concept)等进行分类

        # 句子节点可以通过fully_supported_sentence_nodes和evidence_note获知
        # 首先匹配问题节点
        all_query_phrases = set()
        all_evidence_phrases = set()
        all_fully_supported_evidence_phrases = set()  # 从fully_supported_sentence中提取的短语
        all_partially_supported_evidence_phrases = set()
        evidence_phrase2sentence_nodes = defaultdict(set)  # 记录phrase对应的句子
        add_phrases_info_dict = defaultdict(set)  # 记录要新增的边phrase-sentence信息

        for solution in batch_query_solutions:
            all_query_phrases.update(solution.query_phrases)
            evidence_note = solution.evidence_note
            for sent_node, evidence in evidence_note.items():
                keywords = evidence['keywords']
                if sent_node in solution.fully_supported_sentence_nodes:
                    all_fully_supported_evidence_phrases.update(keywords)
                else:
                    all_partially_supported_evidence_phrases.update(keywords)
                for phrase in keywords:
                    evidence_phrase2sentence_nodes[phrase].add(sent_node)

        all_evidence_phrases.update(all_fully_supported_evidence_phrases | all_partially_supported_evidence_phrases)

        query_entity_phrases, query_concept_phrases, query_phrase_to_same_phrase_nodes, query_phrase_to_similar_phrase_nodes = self.match_relevant_phrase_nodes(
            list(all_query_phrases))
        # 接下来，从图中匹配filter提取出来的证据级phrase相关节点
        evidence_entity_phrases, evidence_concept_phrases, evidence_phrase_to_same_phrase_nodes, evidence_phrase_to_similar_phrase_nodes = self.match_relevant_phrase_nodes(
            list(all_evidence_phrases))
        for solution in batch_query_solutions:
            # 首先定位图中与问题相关的phrase节点
            query_phrases = solution.query_phrases

            query_entity_phrase_nodes = {'same': set(), 'similar': set()}
            query_concept_phrase_nodes = {'same': set(), 'similar': set()}
            for phrase in query_phrases:
                same_phrase_nodes = query_phrase_to_same_phrase_nodes[phrase]
                similar_phrase_nodes = query_phrase_to_similar_phrase_nodes[phrase]

                if phrase in query_entity_phrases:
                    query_entity_phrase_nodes['same'].update(same_phrase_nodes)
                    query_entity_phrase_nodes['similar'].update(similar_phrase_nodes)
                else:
                    query_concept_phrase_nodes['same'].update(same_phrase_nodes)
                    query_concept_phrase_nodes['similar'].update(similar_phrase_nodes)

            # 去掉交集，每种独立
            query_entity_phrase_nodes['same'] -= query_concept_phrase_nodes['same']  # 因为entity有的概念是大写词也包含在内了，所以减去概念词汇
            set_same_phrase_nodes = query_entity_phrase_nodes['same'] | query_concept_phrase_nodes['same']
            query_concept_phrase_nodes['similar'] -= set_same_phrase_nodes
            query_entity_phrase_nodes['similar'] -= set_same_phrase_nodes | query_concept_phrase_nodes['similar']

            phrase_node2score = dict()
            phrase_node2type = dict()
            unevidence_phrase_node2score = dict()
            unevidence_phrase_node2type = dict()
            for sent_node, evidence in solution.evidence_note.items():
                keywords = evidence['keywords']
                score = evidence['score']
                sent_relevant_phrase_nodes = self.node_sentence2phrases[sent_node]
                sent_evidence_phrase_nodes = set()
                for phrase in keywords:
                    if phrase in evidence_entity_phrases:
                        phrase_type = 1      # concept
                    else:
                        phrase_type = 0      # entity
                    graph_phrase_nodes = evidence_phrase_to_same_phrase_nodes[phrase]
                    sent_evidence_phrase_nodes.update(graph_phrase_nodes)
                    for phrase_node in graph_phrase_nodes:
                        if phrase_node not in phrase_node2type:
                            phrase_node2type[phrase_node] = 0
                        if phrase_node not in phrase_node2score:
                            phrase_node2score[phrase_node] = 0

                        phrase_node2type[phrase_node] = max(phrase_node2type[phrase_node], phrase_type)
                        phrase_node2score[phrase_node] = max(phrase_node2score[phrase_node], score)

                unevidence_phrase_nodes = list(sent_relevant_phrase_nodes - sent_evidence_phrase_nodes)
                unevidence_phrase_texts = self.np_graph_texts[unevidence_phrase_nodes]
                for phrase_node, text in zip(unevidence_phrase_nodes, unevidence_phrase_texts):
                    if has_uppercase(text) or has_number(text):
                        phrase_type = 0
                    else:
                        phrase_type = 1
                    if phrase_node not in unevidence_phrase_node2score:
                        unevidence_phrase_node2score[phrase_node] = 0
                    if phrase_node not in unevidence_phrase_node2type:
                        unevidence_phrase_node2type[phrase_node] = 0
                    unevidence_phrase_node2type[phrase_node] = max(unevidence_phrase_node2type[phrase_node], phrase_type)
                    unevidence_phrase_node2score[phrase_node] = max(unevidence_phrase_node2score[phrase_node], score)



            cur_evidence_entity_phrase_nodes = []
            cur_evidence_entity_phrase_node_scores = []
            cur_evidence_concept_phrase_nodes = []
            cur_evidence_concept_phrase_node_scores = []
            for node, phrase_type in phrase_node2type.items():
                if phrase_type == 1:
                    cur_evidence_concept_phrase_nodes.append(node)
                    cur_evidence_concept_phrase_node_scores.append(phrase_node2score[node])
                else:
                    cur_evidence_entity_phrase_nodes.append(node)
                    cur_evidence_entity_phrase_node_scores.append(phrase_node2score[node])

            cur_unevidence_entity_phrase_nodes = []
            cur_unevidence_entity_phrase_node_scores = []
            cur_unevidence_concept_phrase_nodes = []
            cur_unevidence_concept_phrase_node_scores = []
            for node, phrase_type in unevidence_phrase_node2type.items():
                if phrase_type == 1:
                    cur_unevidence_concept_phrase_nodes.append(node)
                    cur_unevidence_concept_phrase_node_scores.append(unevidence_phrase_node2score[node])
                else:
                    cur_unevidence_entity_phrase_nodes.append(node)
                    cur_unevidence_entity_phrase_node_scores.append(unevidence_phrase_node2score[node])


            solution.query_entity_phrase_nodes = query_entity_phrase_nodes
            solution.query_concept_phrase_nodes = query_concept_phrase_nodes
            solution.evidence_entity_phrase_nodes = np.array(cur_evidence_entity_phrase_nodes)
            solution.evidence_concept_phrase_nodes = np.array(cur_evidence_concept_phrase_nodes)
            solution.unevidence_entity_phrase_nodes = np.array(cur_unevidence_entity_phrase_nodes)

            solution.graph_phrase_scores = np.zeros(self.graph.vcount())
            solution.graph_phrase_factors = np.zeros(self.graph.vcount())
            solution.graph_phrase_scores[list(solution.evidence_entity_phrase_nodes)] += np.array(cur_evidence_entity_phrase_node_scores)
            solution.graph_phrase_factors[list(solution.evidence_entity_phrase_nodes)] = self.config.shallow_memory.sentence_phrase_factor * self.config.shallow_memory.entity_phrase_factor *  self.config.shallow_memory.filtered_phrase_factor * self.config.shallow_memory.phrase_ratio
            solution.graph_phrase_scores[list(solution.evidence_concept_phrase_nodes)] += np.array(cur_evidence_concept_phrase_node_scores)
            solution.graph_phrase_factors[list(solution.evidence_concept_phrase_nodes)] = self.config.shallow_memory.sentence_phrase_factor * self.config.shallow_memory.concept_phrase_factor * self.config.shallow_memory.filtered_phrase_factor * self.config.shallow_memory.phrase_ratio
            # 给evidence sentence中未筛选上的phrase赋权
            solution.graph_phrase_scores[list(solution.unevidence_entity_phrase_nodes)] += np.array(cur_unevidence_entity_phrase_node_scores)
            solution.graph_phrase_factors[list(solution.unevidence_entity_phrase_nodes)] = self.config.shallow_memory.sentence_phrase_factor * self.config.shallow_memory.entity_phrase_factor * self.config.shallow_memory.unfiltered_phrase_factor * self.config.shallow_memory.phrase_ratio

            solution.evidence_entity_phrases = self.np_graph_texts[list(solution.evidence_entity_phrase_nodes)]
            solution.evidence_concept_phrases = self.np_graph_texts[list(solution.evidence_concept_phrase_nodes)]
            solution.unevidence_entity_phrases = self.np_graph_texts[list(solution.unevidence_entity_phrase_nodes)]
            solution.query_entity_phrases = list(query_entity_phrase_nodes['same'])
            solution.query_concept_phrases = list(query_concept_phrase_nodes['same'])
        return add_phrases_info_dict
    def assign_graph_weights(self, solution):
        # 没有提取的证据，则不会有图信息
        if len(solution.evidence_note) == 0:
            return solution

        if solution.global_graph_scores is None:
            node_num = self.graph.vcount()
            global_graph_scores = np.zeros(node_num)
        else:
            global_graph_scores = solution.global_graph_scores

        # 赋权前，要判断哪些文档之间出现过，但没有提取过有用信息
        # 找到哪些从未被确定为evidence的句子
        all_processed_sentence_nodes = set()
        for doc_node in solution.processed_document_nodes:
            sent_nodes = self.node_doc2sentences[doc_node]
            all_processed_sentence_nodes.update(sent_nodes)
        valueless_sentence_nodes = all_processed_sentence_nodes - set(solution.evidence_note.keys())

        # 首先缩小原始权重分布
        global_graph_scores *= 0.01
        # 降低无价值的句子权重
        if len(valueless_sentence_nodes):
            global_graph_scores[list(valueless_sentence_nodes)] /= 2

        # 给文档赋权
        if self.config.debug.enable_weight_doc_node:
            global_graph_scores[solution.ranked_doc_nodes] += np.array(solution.ranked_doc_scores) * self.config.shallow_memory.document_ratio

        # 给句子赋权
        supported_sentence_nodes = []
        supported_sentence_scores = []
        for evidence in solution.evidence_note.values():
            node = evidence['id']
            score = evidence['score']
            supported_sentence_nodes.append(node)
            supported_sentence_scores.append(score)

        if self.config.debug.enable_weight_sentence_node:
            global_graph_scores[supported_sentence_nodes] += np.array(supported_sentence_scores) * self.config.shallow_memory.sentence_ratio

        # 给Phrase赋权
        if self.config.debug.enable_weight_phrase_node:
            global_graph_scores += self.node_idfs * solution.graph_phrase_scores * solution.graph_phrase_factors

        # 给问题phrase赋权
        query_base_score = max(supported_sentence_scores)
        if self.config.debug.enable_weight_query_phrase_node:
            query_entity_phrase_nodes = list(solution.query_entity_phrase_nodes['same'] - set(solution.evidence_entity_phrase_nodes))
            query_concept_phrase_nodes = list(solution.query_concept_phrase_nodes['same'] - set(solution.evidence_concept_phrase_nodes))
            global_graph_scores[query_entity_phrase_nodes] += self.node_idfs[query_entity_phrase_nodes] * query_base_score * self.config.shallow_memory.query_phrase_factor * self.config.shallow_memory.entity_phrase_factor * self.config.shallow_memory.phrase_ratio
            global_graph_scores[query_concept_phrase_nodes] += self.node_idfs[query_concept_phrase_nodes] * query_base_score * self.config.shallow_memory.query_phrase_factor * self.config.shallow_memory.concept_phrase_factor * self.config.shallow_memory.phrase_ratio

        solution.global_graph_scores, solution.ppr_graph_scores = self.convert_score_global2local(global_graph_scores)

        return solution
    def batch_assign_graph_weights(self, batch_query_solutions):
        for query_solution in batch_query_solutions:
            self.assign_graph_weights(query_solution)

        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     futures = [executor.submit(self.assign_graph_weights, query_solution) for query_solution in
        #                batch_query_solutions]
        #     batch_query_solutions = [future.result() for future in futures]
        return batch_query_solutions

    def select_doc_node_by_sentence_nodes(self, sentence_nodes, top_k):
        # 根据sentence取文档
        selected_doc_node = []
        all_doc_nodes = set(self.set_doc_nodes)
        for sentence in sentence_nodes:
            neighbors = self.graph.neighbors(sentence)
            doc_nodes = set(neighbors) & all_doc_nodes
            for node in doc_nodes:
                if node not in selected_doc_node:
                    selected_doc_node.append(node)

            if len(selected_doc_node) >= top_k:
                break

        return selected_doc_node

    def get_sorted_docs(self, global_graph_scores, top_k):
        # 获取排好序的文档
        # 判断当前PPR图类型
        sentence_nodes = list(self.set_sentence_nodes)
        doc_nodes = list(self.set_doc_nodes)
        if self.ppr_structure == 's2e':
            sentence_scores = global_graph_scores[sentence_nodes]
            sorted_idx = np.argsort(sentence_scores)[::-1]
            sorted_sentence_nodes = np.array(sentence_nodes)[sorted_idx]
            sorted_doc_nodes = self.select_doc_node_by_sentence_nodes(sorted_sentence_nodes, top_k)
        else:  # d2s2e
            doc_scores = global_graph_scores[doc_nodes]
            sorted_idx = np.argsort(doc_scores)[::-1]
            # 得到排序后的原始node_id
            sorted_doc_nodes = np.array(doc_nodes)[sorted_idx][:top_k]

        sorted_docs = self.np_graph_texts[sorted_doc_nodes]
        sorted_doc_keys = self.np_graph_keys[sorted_doc_nodes]
        sorted_doc_scores = global_graph_scores[sorted_doc_nodes]

        return sorted_docs, sorted_doc_keys, sorted_doc_scores

    def convert_score_global2local(self, old_global_graph_scores):
        # 由于图更新了，所有之前的图分数可能确实部分节点
        global_graph_scores = np.zeros(self.graph.vcount())
        # 新增的节点都是按照顺序，往后追加节点
        old_global_nodes = list(range(old_global_graph_scores.size))
        global_graph_scores[old_global_nodes] += old_global_graph_scores


        local_graph_scores = global_graph_scores[self.local2global_nodes]
        return global_graph_scores, local_graph_scores

    def convert_score_local2global(self, global_graph_scores, local_graph_scores):
        # 如何将局部映射到全局
        global_graph_scores[self.local2global_nodes] = local_graph_scores

        return global_graph_scores

    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5):
        if damping is None: damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.ppr_graph.personalized_pagerank(
            vertices=range(self.ppr_graph.vcount()),
            damping=damping,
            directed=False,
            reset=reset_prob,
            implementation='prpack'
        )
        pagerank_scores = np.array(pagerank_scores)
        return pagerank_scores


    def batch_ppr(self, batch_query_solutions):
        for solution in tqdm(batch_query_solutions, desc="PPR"):
            if len(solution.evidence_note) == 0:
                solution.reranked_docs = solution.ranked_docs
                solution.reranked_doc_keys = solution.ranked_doc_keys
                solution.reranked_doc_scores = solution.ranked_doc_scores
                solution.history_iter_prr_reranked_docs.append(solution.ranked_docs[:self.config.rag.final_search_doc_num])
                continue
            # 由于PPR不是在整个图上运行，可能只在部分类型节点上运行，需要取出子图
            # try:
            if self.config.debug.enable_graph_reranker:
                solution.ppr_graph_scores = self.run_ppr(solution.ppr_graph_scores, self.config.rag.damping)
            # 局部转为全局分数
            solution.global_graph_scores = self.convert_score_local2global(solution.global_graph_scores, solution.ppr_graph_scores)
            sorted_docs, sorted_doc_keys, sorted_doc_scores = self.get_sorted_docs(solution.global_graph_scores, self.config.rag.search_doc_num)

            solution.reranked_docs = sorted_docs
            solution.reranked_doc_keys = sorted_doc_keys
            solution.reranked_doc_scores = sorted_doc_scores
            solution.history_iter_prr_reranked_docs.append(sorted_docs[:self.config.rag.final_search_doc_num])

    def batch_retrieve(self, batch_query_solutions):
        batch_queries = []
        for solution in batch_query_solutions:
            batch_queries.append(solution.augment_query)
        batch_all_documents = self.retriever.batch_search(batch_queries, top_k=int(self.config.rag.search_doc_num*2), search_mode='query_to_doc')
        # total = 0
        for solution, documents in zip(batch_query_solutions, batch_all_documents):
            ranked_docs = []
            ranked_doc_scores = []
            ranked_doc_keys = []
            ranked_doc_nodes = []
            for document in documents:
                text = document['content']
                key = document['key']
                score = document['score']
                node = self.key2node.get(key, None)
                if node is None:
                    print("错误节点: ", key)
                    continue
                ranked_docs.append(text)
                ranked_doc_scores.append(score)
                ranked_doc_keys.append(key)
                ranked_doc_nodes.append(node)
            solution.ranked_docs = ranked_docs
            solution.ranked_doc_scores = np.array(ranked_doc_scores) * self.node_idfs[ranked_doc_nodes]# / self.all_degrees[ranked_doc_nodes]#compute_positive_rrf_scores(np.array(ranked_doc_scores))
            solution.ranked_doc_keys = ranked_doc_keys
            solution.ranked_doc_nodes = ranked_doc_nodes

            solution.history_iter_pre_ppr_reranked_docs.append(ranked_docs[:self.config.rag.final_search_doc_num])
            # total += len(ranked_doc_nodes)

        # print("shallow 检索数量: ", total)

    def run(self, batch_query_solutions):
        # Step1: Filter!
        logger.info(
            f"Subqueries: Total: {self.config.debug.enable_query_decompose}, Subqueris:{self.config.debug.enable_subquery}, rewrite: {self.config.debug.enable_rewrite_query}, cloze: {self.config.debug.enable_cloze}")
        logger.info("Step4.1: Filter Knowledges ...")
        self.filter_knowledges(batch_query_solutions)

        # Step2: Augment Query!
        logger.info(f"Step4.2: Knowledges Augment Query ... [{self.config.debug.enable_query_augment}]")
        if self.config.debug.enable_query_augment:
            self.query_optimizer.batch_augment(batch_query_solutions)
        else:
            for solution in batch_query_solutions:
                solution.augment_query = solution.query

        # Step3: Retriever Query
        logger.info("Step4.3: Retrieve Augment Query[Evidence-based Re-Rank] ...")
        self.batch_retrieve(batch_query_solutions)

        # Step4: 节点匹配+节点分类
        logger.info("Step4.4: Node Matching & Classify ...")
        add_phrases_info_dict = self.classify_phrase_nodes(batch_query_solutions)

        # Step5: 更新图
        logger.info("Step4.5: Updating Memory with New Phrases ...")

        self.update_shallow_memory_with_phrases(add_phrases_info_dict)

        # Step6: 节点赋权
        logger.info("Step4.6: Node Weighting...")
        logger.info(f"passage: {self.config.debug.enable_weight_doc_node}, sentence: {self.config.debug.enable_weight_sentence_node}, phrase: {self.config.debug.enable_weight_phrase_node}, query:{self.config.debug.enable_weight_query_phrase_node}")
        batch_query_solutions = self.batch_assign_graph_weights(batch_query_solutions)

        # Step7: PPR
        logger.info(f"Step4.7: PPR[Context-based Re-rank] ...[{self.config.debug.enable_graph_reranker}]")
        self.batch_ppr(batch_query_solutions)

