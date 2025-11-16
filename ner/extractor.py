'''
功能简介：
给定文档，去抽取信息，返回文本，句子，片段，实体
'''
from typing import List
from utils.file_manager import load_json
from utils.cache_manager import save_cache, get_cache_keys, load_cache, save_parquet, load_parquet
from utils.text_manager import generate_key, has_uppercase, has_number
import spacy, os
from ner.spacy_ner import get_sentences, ner_all_keywords
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.log_manager import get_logger
import numpy as np
logger = get_logger(__name__)

'''
目录结构
- output_root:
    - graph_rag_model
    - extractor
        - spacy_model: en_core_web_sm
            - 缓存文件
    - retriever
        - embedding_name
            - vector_store
            - cache
    - info_filter
        - LLM_name
    - decomposer
        - LLM_name
'''
class Extractor:
    def __init__(self, output_root: str, max_workers: int = 100, spacy_model='en_core_web_sm', enable_ner_tqdm=True):
        self.nlp = spacy.load(spacy_model)
        self.enable_ner_tqdm = enable_ner_tqdm
        self.max_workers = max_workers
        self.cache_root = os.path.join(output_root, 'extractor', spacy_model)
        os.makedirs(self.cache_root, exist_ok=True)

    def ner_corpus_to_sqlite(self, corpus_path):
        '''
        缓存:
        doc ->  sentence_keys,  entites
        sentence -> phrases, fragment 片段.

        希望通过doc, 获取所有句子和实体  T   句子按顺序存放
        通过sentence， 获取所有片段和实体  T
        :return:
        '''
        corpus = load_json(corpus_path)

        if os.path.exists(os.path.join(self.cache_root, 'cache.sqlite')):
            logger.info("Load Extractor Cache ...")
            all_doc_keys = get_cache_keys(self.cache_root, filename='cache', table_name='doc')
            batch_ner_info = load_cache(self.cache_root, all_doc_keys, filename='cache', table_name='doc')
        else:
            batch_docs = []
            batch_titles = []
            # 然后调用spacy进行切割
            for item in corpus:
                text = item['text']
                title = item['title']
                batch_docs.append(f'{title}\n\n{text}')
                batch_titles.append(title)
            # 抽取信息s
            batch_ner_info = self.batch_ner_doc(batch_docs, batch_titles)
            logger.info(f"Save Ner Corpus info to {os.path.join(self.cache_root, 'cache.sqlite')} ...")
            save_cache(self.cache_root, batch_ner_info, filename='cache', table_name='doc')

        corpus_doc_info = {'key': [],
                           'content': [],
                           'sentence_keys': [],
                           'phrase_keys': []}

        corpus_sentence_info = {'key': [],
                                'content': [],
                                'phrase_keys': []}

        corpus_phrase_info = {'key': [],
                              'content': []}

        for ner_info in batch_ner_info:
            doc_key = ner_info['key']
            if doc_key not in corpus_doc_info['key']:
                corpus_doc_info['key'].append(doc_key)
                corpus_doc_info['content'].append(ner_info['content'])
                corpus_doc_info['sentence_keys'].append(ner_info['sentence_keys'])
                corpus_doc_info['phrase_keys'].append(ner_info['phrase_keys'])

            unprocessed_sentence_indices = []
            for idx, sentence_key in enumerate(ner_info['sentence_keys']):
                if sentence_key in corpus_sentence_info['key']:
                    continue
                unprocessed_sentence_indices.append(idx)
            sentence_keys = np.array(ner_info['sentence_keys'])
            unprocessed_sentence_keys = sentence_keys[unprocessed_sentence_indices].tolist()

            sentence_texts = np.array(ner_info['sentence_texts'])
            unprocessed_sentence_texts = sentence_texts[unprocessed_sentence_indices].tolist()

            sentence2phrase_keys = ner_info['sentence2phrase_keys']
            unprocessed_sentence2phrase_keys = []
            for index in unprocessed_sentence_indices:
                unprocessed_sentence2phrase_keys.append(sentence2phrase_keys[index])

            corpus_sentence_info['key'].extend(unprocessed_sentence_keys)
            corpus_sentence_info['content'].extend(unprocessed_sentence_texts)
            corpus_sentence_info['phrase_keys'].extend(unprocessed_sentence2phrase_keys)

            unprocessed_phrase_indices = []
            for idx, phrase_key in enumerate(ner_info['phrase_keys']):
                if phrase_key in corpus_phrase_info['key']:
                    continue
                unprocessed_phrase_indices.append(idx)
            phrase_keys = np.array(ner_info['phrase_keys'])
            unprocessed_phrase_keys = phrase_keys[unprocessed_phrase_indices].tolist()
            phrase_texts = np.array(ner_info['phrase_texts'])
            unprocessed_phrase_texts = phrase_texts[unprocessed_phrase_indices].tolist()

            corpus_phrase_info['key'].extend(unprocessed_phrase_keys)
            corpus_phrase_info['content'].extend(unprocessed_phrase_texts)


        return corpus_doc_info, corpus_sentence_info, corpus_phrase_info


    def ner_documents_to_sqlite(self, texts):
        '''
        缓存:
        doc ->  sentence_keys,  entites
        sentence -> phrases, fragment 片段.

        希望通过doc, 获取所有句子和实体  T   句子按顺序存放
        通过sentence， 获取所有片段和实体  T
        :return:
        '''
        sequence_docs = []
        sequence_tiltes = []
        batch_keys = []
        all_doc_keys = get_cache_keys(self.cache_root, filename='cache', table_name='doc')
        for text in texts:
            # text = item['content']
            # title = item['title']
            # key = item['key']
            key = self.get_doc_key(text)
            if key == 'doc-9ae04f2a2a4e921561664baa437c2e9afad1648b73cfc83aceed8d31ec6f1aa9':
                print("fack::::")
            batch_keys.append(key)
            if key in all_doc_keys:
                if key == 'doc-9ae04f2a2a4e921561664baa437c2e9afad1648b73cfc83aceed8d31ec6f1aa9':
                    print("???", key)
                continue
            sequence_docs.append(text)
            sequence_tiltes.append(key)

        # 抽取信息s
        batch_ner_info = self.batch_ner_doc(sequence_docs, sequence_tiltes)
        if len(batch_ner_info):
            logger.info(f"Save Ner Corpus info to {os.path.join(self.cache_root, 'cache.sqlite')} ...")
            save_cache(self.cache_root, batch_ner_info, filename='cache', table_name='doc')
        logger.info(f"Read Ner Corpus info to {os.path.join(self.cache_root, 'cache.sqlite')} ...")
        batch_ner_info = load_cache(self.cache_root, batch_keys, filename='cache', table_name='doc')


        corpus_doc_info = {'key': [],
                           'content': [],
                           'sentence_keys': [],
                           'phrase_keys': []}

        corpus_sentence_info = {'key': [],
                                'content': [],
                                'phrase_keys': []}

        corpus_phrase_info = {'key': [],
                              'content': []}

        for ner_info in batch_ner_info:
            doc_key = ner_info['key']
            if doc_key not in corpus_doc_info['key']:
                corpus_doc_info['key'].append(doc_key)
                corpus_doc_info['content'].append(ner_info['content'])
                corpus_doc_info['sentence_keys'].append(ner_info['sentence_keys'])
                corpus_doc_info['phrase_keys'].append(ner_info['phrase_keys'])

            unprocessed_sentence_indices = []
            for idx, sentence_key in enumerate(ner_info['sentence_keys']):
                if sentence_key in corpus_sentence_info['key']:
                    continue
                unprocessed_sentence_indices.append(idx)
            sentence_keys = np.array(ner_info['sentence_keys'])
            unprocessed_sentence_keys = sentence_keys[unprocessed_sentence_indices].tolist()

            sentence_texts = np.array(ner_info['sentence_texts'])
            unprocessed_sentence_texts = sentence_texts[unprocessed_sentence_indices].tolist()

            sentence2phrase_keys = ner_info['sentence2phrase_keys']
            unprocessed_sentence2phrase_keys = []
            for index in unprocessed_sentence_indices:
                unprocessed_sentence2phrase_keys.append(sentence2phrase_keys[index])

            corpus_sentence_info['key'].extend(unprocessed_sentence_keys)
            corpus_sentence_info['content'].extend(unprocessed_sentence_texts)
            corpus_sentence_info['phrase_keys'].extend(unprocessed_sentence2phrase_keys)

            unprocessed_phrase_indices = []
            for idx, phrase_key in enumerate(ner_info['phrase_keys']):
                if phrase_key in corpus_phrase_info['key']:
                    continue
                unprocessed_phrase_indices.append(idx)
            phrase_keys = np.array(ner_info['phrase_keys'])
            unprocessed_phrase_keys = phrase_keys[unprocessed_phrase_indices].tolist()
            phrase_texts = np.array(ner_info['phrase_texts'])
            unprocessed_phrase_texts = phrase_texts[unprocessed_phrase_indices].tolist()

            corpus_phrase_info['key'].extend(unprocessed_phrase_keys)
            corpus_phrase_info['content'].extend(unprocessed_phrase_texts)


        return corpus_doc_info, corpus_sentence_info, corpus_phrase_info

    def ner_corpus_to_parquet(self, corpus_path):
        doc_file = os.path.join(self.cache_root, 'docs.parquet')
        sentence_file = os.path.join(self.cache_root, 'sentences.parquet')
        phrase_file = os.path.join(self.cache_root, 'phrases.parquet')
        if os.path.exists(doc_file) and os.path.exists(sentence_file) and os.path.exists(phrase_file):
            # 加载缓存
            corpus_doc_info = load_parquet(self.cache_root, 'docs')
            corpus_sentence_info = load_parquet(self.cache_root, 'sentences')
            corpus_phrase_info = load_parquet(self.cache_root, 'phrases')
        else:
            corpus = load_json(corpus_path)
            batch_docs = []
            batch_titles = []
            # 然后调用spacy进行切割
            for item in corpus:
                text = item['text']
                title = item['title']
                batch_docs.append(f'{title}\n\n{text}')
                batch_titles.append(title)
            # 抽取信息
            batch_ner_info = self.batch_ner_doc(batch_docs, batch_titles)
            corpus_doc_info = {'key': [],
                               'content': [],
                               'sentence_keys': [],
                               'phrase_keys': []}

            corpus_sentence_info = {'key': [],
                                    'content': [],
                                    'phrase_keys': []}

            corpus_phrase_info = {'key': [],
                                  'content': []}
            logger.info("Save Ner Corpus Info...")
            for ner_info in batch_ner_info:
                doc_key = ner_info['key']
                if doc_key not in corpus_doc_info['key']:
                    corpus_doc_info['key'].append(doc_key)
                    corpus_doc_info['content'].append(ner_info['content'])
                    corpus_doc_info['sentence_keys'].append(ner_info['sentence_keys'])
                    corpus_doc_info['phrase_keys'].append(ner_info['phrase_keys'])

                unprocessed_sentence_indices = []
                for idx, sentence_key in enumerate(ner_info['sentence_keys']):
                    if sentence_key in corpus_sentence_info['key']:
                        continue
                    unprocessed_sentence_indices.append(idx)
                sentence_keys = np.array(ner_info['sentence_keys'])
                unprocessed_sentence_keys = sentence_keys[unprocessed_sentence_indices].tolist()

                sentence_texts = np.array(ner_info['sentence_texts'])
                unprocessed_sentence_texts = sentence_texts[unprocessed_sentence_indices].tolist()

                sentence2phrase_keys = ner_info['sentence2phrase_keys']
                unprocessed_sentence2phrase_keys = []
                for index in unprocessed_sentence_indices:
                    unprocessed_sentence2phrase_keys.append(sentence2phrase_keys[index])

                corpus_sentence_info['key'].extend(unprocessed_sentence_keys)
                corpus_sentence_info['content'].extend(unprocessed_sentence_texts)
                corpus_sentence_info['phrase_keys'].extend(unprocessed_sentence2phrase_keys)

                unprocessed_phrase_indices = []
                for idx, phrase_key in enumerate(ner_info['phrase_keys']):
                    if phrase_key in corpus_phrase_info['key']:
                        continue
                    unprocessed_phrase_indices.append(idx)
                phrase_keys = np.array(ner_info['phrase_keys'])
                unprocessed_phrase_keys = phrase_keys[unprocessed_phrase_indices].tolist()
                phrase_texts = np.array(ner_info['phrase_texts'])
                unprocessed_phrase_texts = phrase_texts[unprocessed_phrase_indices].tolist()

                corpus_phrase_info['key'].extend(unprocessed_phrase_keys)
                corpus_phrase_info['content'].extend(unprocessed_phrase_texts)

            save_parquet(self.cache_root, corpus_doc_info, 'docs')
            save_parquet(self.cache_root, corpus_sentence_info, 'sentences')
            save_parquet(self.cache_root, corpus_phrase_info, 'phrases')

        return corpus_doc_info, corpus_sentence_info, corpus_phrase_info


    def ner_documents_to_parquet(self, texts):
        doc_file = os.path.join(self.cache_root, 'docs.parquet')
        sentence_file = os.path.join(self.cache_root, 'sentences.parquet')
        phrase_file = os.path.join(self.cache_root, 'phrases.parquet')
        if os.path.exists(doc_file):
            # 加载缓存
            corpus_doc_info = load_parquet(self.cache_root, 'docs')
        else:
            corpus_doc_info = {'key': [],
                               'content': [],
                               'sentence_keys': [],
                               'phrase_keys': []}
        if os.path.exists(sentence_file) and os.path.exists(phrase_file):
            corpus_sentence_info = load_parquet(self.cache_root, 'sentences')
        else:
            corpus_sentence_info = {'key': [],
                                    'content': [],
                                    'phrase_keys': []}
        if os.path.exists(phrase_file):
            corpus_phrase_info = load_parquet(self.cache_root, 'phrases')
        else:
            corpus_phrase_info = {'key': [],
                                  'content': []}
        allowed_texts = set(texts) - set(corpus_doc_info['content'])
        batch_docs = []
        batch_titles = []
        cache_keys = []
        # 然后调用spacy进行切割
        for text in texts:
            key = self.get_doc_key(text)
            if text in allowed_texts:
                cache_keys.append(key)
                continue
            batch_docs.append(text)
            batch_titles.append(key)

        # 抽取信息
        batch_ner_info = self.batch_ner_doc(batch_docs, batch_titles)
        logger.info("Save Ner Texts Info...")

        ner_doc_info = {'key': [],
                        'content': [],
                        'sentence_keys': [],
                        'phrase_keys': []}

        ner_sentence_info = {'key': [],
                             'content': [],
                             'phrase_keys': []}

        ner_phrase_info = {'key': [],
                           'content': []}
        for ner_info in batch_ner_info:
            doc_key = ner_info['key']
            if doc_key not in corpus_doc_info['key']:
                # corpus_doc_info['key'].append(doc_key)
                # corpus_doc_info['content'].append(ner_info['content'])
                # corpus_doc_info['sentence_keys'].append(ner_info['sentence_keys'])
                # corpus_doc_info['phrase_keys'].append(ner_info['phrase_keys'])
                ner_doc_info['key'].append(doc_key)
                ner_doc_info['content'].append(ner_info['content'])
                ner_doc_info['sentence_keys'].append(ner_info['sentence_keys'])
                ner_doc_info['phrase_keys'].append(ner_info['phrase_keys'])


            unprocessed_sentence_indices = []
            for idx, sentence_key in enumerate(ner_info['sentence_keys']):
                if sentence_key in ner_sentence_info['key']:
                    continue
                unprocessed_sentence_indices.append(idx)
            sentence_keys = np.array(ner_info['sentence_keys'])
            unprocessed_sentence_keys = sentence_keys[unprocessed_sentence_indices].tolist()

            sentence_texts = np.array(ner_info['sentence_texts'])
            unprocessed_sentence_texts = sentence_texts[unprocessed_sentence_indices].tolist()

            sentence2phrase_keys = ner_info['sentence2phrase_keys']
            unprocessed_sentence2phrase_keys = []
            for index in unprocessed_sentence_indices:
                unprocessed_sentence2phrase_keys.append(sentence2phrase_keys[index])

            ner_sentence_info['key'].extend(unprocessed_sentence_keys)
            ner_sentence_info['content'].extend(unprocessed_sentence_texts)
            ner_sentence_info['phrase_keys'].extend(unprocessed_sentence2phrase_keys)

            unprocessed_phrase_indices = []
            for idx, phrase_key in enumerate(ner_info['phrase_keys']):
                if phrase_key in ner_phrase_info['key']:
                    continue
                unprocessed_phrase_indices.append(idx)
            phrase_keys = np.array(ner_info['phrase_keys'])
            unprocessed_phrase_keys = phrase_keys[unprocessed_phrase_indices].tolist()
            phrase_texts = np.array(ner_info['phrase_texts'])
            unprocessed_phrase_texts = phrase_texts[unprocessed_phrase_indices].tolist()

            ner_phrase_info['key'].extend(unprocessed_phrase_keys)
            ner_phrase_info['content'].extend(unprocessed_phrase_texts)

        if len(allowed_texts):
            corpus_doc_info['key'] += ner_doc_info['key']
            corpus_doc_info['content'] += ner_doc_info['content']
            corpus_doc_info['sentence_keys'] += ner_doc_info['sentence_keys']
            corpus_doc_info['phrase_keys'] += ner_doc_info['phrase_keys']

            corpus_sentence_info['key'] += ner_sentence_info['key']
            corpus_sentence_info['content'] += ner_sentence_info['content']
            corpus_sentence_info['phrase_keys'] += ner_sentence_info['phrase_keys']

            corpus_phrase_info['key'] += ner_phrase_info['key']
            corpus_phrase_info['content'] += ner_phrase_info['content']

            save_parquet(self.cache_root, corpus_doc_info, 'docs')
            save_parquet(self.cache_root, corpus_sentence_info, 'sentences')
            save_parquet(self.cache_root, corpus_phrase_info, 'phrases')

        # 接着根据keys取出数据
        cache_sentence_keys = set()
        cache_phrase_keys = set()
        for idx, key in enumerate(corpus_doc_info['key']):
            if key in cache_keys:
                content = corpus_doc_info['content'][idx]
                sentence_keys = corpus_sentence_info['sentence_keys'][idx]
                phrase_keys = corpus_phrase_info['phrase_keys'][idx]

                ner_doc_info['key'].append(key)
                ner_doc_info['content'].append(content)
                ner_doc_info['sentence_keys'].append(sentence_keys)
                ner_doc_info['phrase_keys'].append(phrase_keys)

                cache_sentence_keys.update(sentence_keys)
                cache_phrase_keys.update(phrase_keys)
        for idx, key in enumerate(corpus_sentence_info['key']):
            if key in cache_sentence_keys:
                content = corpus_sentence_info['content'][idx]
                phrase_keys = corpus_phrase_info['phrase_keys'][idx]

                ner_sentence_info['key'].append(key)
                ner_sentence_info['content'].append(content)
                ner_sentence_info['phrase_keys'].append(phrase_keys)

        for idx, key in enumerate(corpus_phrase_info['key']):
            if key in cache_phrase_keys:
                content = corpus_phrase_info['content'][idx]
                ner_phrase_info['key'].append(key)
                ner_phrase_info['content'].append(content)


        return ner_doc_info, ner_sentence_info, ner_phrase_info

    def get_doc_key(self, text: str):
        return generate_key(text, 'doc-')

    def get_sentence_key(self, text: str):
        return generate_key(text, 'sentence-')

    def get_phrase_key(self, text: str):
        return generate_key(text, 'phrase-')

    def ner_doc(self, text: str, title: str=''):
        # 存储信息
        '''
        doc: {'key', 'content', 'sentence_keys', 'phrase_keys'}
        sentence: {'key', 'content', 'phrase_keys'}
        phrase: {'key', 'content'}
        '''
        # 获取text, sentence, fragment, keyword的信息
        # text = text.strip()
        doc_key = self.get_doc_key(text)

        sequence_sentence_texts = []  # 记录句子
        sequence_sentence_keys = []
        sequence_sentence2phrase_keys = []

        # 句子，片段有序存储，但是关键词无序存储
        key2phrases = {}

        sentences = get_sentences(text)
        nlp_sentences = list(self.nlp.pipe(sentences))

        # 获取句子和片段
        for nlp_sent in nlp_sentences:
            sent_text = nlp_sent.text.strip()
            sent_key = self.get_sentence_key(sent_text)

            # 提取keywords
            # nlp_sent = self.nlp(sent_text)
            # 提取句子的实体
            sentence_all_keywords = ner_all_keywords(nlp_sent)
            cur_key2phrases = {}
            for keyword in sentence_all_keywords:
                phrase_key = self.get_phrase_key(keyword)
                cur_key2phrases[phrase_key] = keyword
            key2phrases.update(cur_key2phrases)
            sentence2phrase_keys = list(cur_key2phrases.keys())

            sequence_sentence_keys.append(sent_key)
            sequence_sentence_texts.append(sent_text)
            sequence_sentence2phrase_keys.append(sentence2phrase_keys)


        sequence_phrase_keys = []
        sequence_phrase_texts = []
        for key, phrase in key2phrases.items():
            sequence_phrase_keys.append(key)
            sequence_phrase_texts.append(phrase)

        ner_info = {'key': doc_key,
                    'title': title,
                    'content': text,
                    'sentence_keys': sequence_sentence_keys,
                    'sentence_texts': sequence_sentence_texts,
                    'sentence2phrase_keys': sequence_sentence2phrase_keys,
                    'phrase_keys': sequence_phrase_keys,
                    'phrase_texts': sequence_phrase_texts}

        return ner_info

    def batch_ner_doc(self, batch_docs: List[str], batch_titles: List[str]):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.ner_doc, text, title) for text, title in zip(batch_docs, batch_titles)]  # 提交任务
            results = [future.result() for future in tqdm(futures, desc="Batch NER Doc", disable=not self.enable_ner_tqdm)]  # 获取所有任务的结果

        return results

    def ner_query(self, query):
        text = query.strip()
        doc = self.nlp(text)
        all_keywords = ner_all_keywords(doc)

        # phrases = []
        # for e in all_keywords:
        #     if has_uppercase(e) or has_number(e):# or len(e.split(' '))>1 or '-' in e:
        #         phrases.append(e)

        return list(all_keywords)

    def batch_ner_text(self, batch_texts):
        nlp_texts = list(self.nlp.pipe(batch_texts))
        batch_keywords = []
        for doc in nlp_texts:
            batch_keywords.append(list(ner_all_keywords(doc)))

        return batch_keywords
    def batch_ner_phrases(self, batch_sequence_phrases):
        batch_texts = []
        for sequence_phrases in batch_sequence_phrases:
            batch_texts.append(', '.join(sequence_phrases))

        return self.batch_ner_text(batch_texts)

    def batch_ner_query(self, batch_queries, enable_tqdm=False):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.ner_query, query) for query in batch_queries]  # 提交任务
            batch_phrases = [future.result() for future in tqdm(futures, desc="Batch NER Query", disable=not enable_tqdm)]  # 获取所有任务的结果

        return batch_phrases
