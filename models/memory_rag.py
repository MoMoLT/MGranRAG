from retrievers.retriever import CacheRetriever
from config import HotpotQAConfig as default_config
from utils.template_manager import TemplateManager
from models.info_filter import SentenceFilter
from models.query_optimizer import QuestionOptimizer
from utils.class_manager import QuerySolution
import numpy as np
import time
from models.shallow_memory import ShallowMemory
from utils.log_manager import get_logger

logger = get_logger(__name__)
'''
目录结构
- output_root:
    - llm_cache
        - llm_name:
            缓存文件
    - shallow_memory
        浅层记忆文件Index Graph
    - deep_memory
        深层记忆文件KG
    - extractor
        - spacy_model: en_core_web_sm
            - 缓存文件
    - retriever
        - embedding_name
            - vector_store
            - cache
'''


class MGranRAG:
    def __init__(self, tokenizer, extractor, embedding_model, base_llm, config=default_config):
        self.config = config

        self.template_manager = TemplateManager()
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.base_retriever = CacheRetriever(embedding_model=embedding_model,
                                             embedding_path=self.config.rag.embedding_path,
                                             output_root=self.config.dataset.output_root,
                                             max_workers=self.config.rag.max_workers,
                                             model_mode=self.config.rag.model_mode,
                                             enable_search_tqdm=self.config.tqdm.enable_search_tqdm)


        self.base_llm = base_llm
        self.query_optimizer = QuestionOptimizer(llm=self.base_llm,
                                             template_manager=self.template_manager,
                                             enable_tqdm=self.config.tqdm.enable_filter_tqdm)
        self.filter = SentenceFilter(llm=self.base_llm,
                                     tokenizer=self.tokenizer,
                                     template_manager=self.template_manager,
                                     enable_tqdm=self.config.tqdm.enable_filter_tqdm,
                                     enable_epoch_filter_single_doc=self.config.debug.enable_epoch_filter_single_doc)

        self.shallow_memory = ShallowMemory(extractor=self.extractor,
                                            base_retriever=self.base_retriever,
                                            filter=self.filter,
                                            query_optimizer=self.query_optimizer,
                                            template_manager=self.template_manager,
                                            remove_data=not self.base_retriever.exist_retriever,
                                            config=self.config)

    def batch_retrieve(self, batch_queries, top_k, search_mode):
        batch_all_documents = self.base_retriever.batch_search(batch_queries, top_k=top_k, search_mode=search_mode)
        batch_document_with_scores = []
        # total = 0
        for documents in batch_all_documents:
            document_with_scores = []
            for document in documents:
                text = document['content']
                key = document['key']
                score = document['score']
                document_with_scores.append((text, key, score))
            batch_document_with_scores.append(document_with_scores)
            # total += len(document_with_scores)
        # print("总检索文档: ", total)
        return batch_document_with_scores

    def batch_ner_query_phrases(self, batch_queries, raw_batch_cloze_phrases):
        # 抽取实体
        batch_query_phrases = []
        batch_ner_query_phrases = self.extractor.batch_ner_query(batch_queries)
        batch_cloze_phrases = self.extractor.batch_ner_phrases(raw_batch_cloze_phrases)
        for ner_query_phrases, cloze_phrases in zip(batch_ner_query_phrases, batch_cloze_phrases):
            ner_query_phrases = set(ner_query_phrases)
            for phrase in cloze_phrases:
                # 有些比不要的需要排除掉， 后续可以添加一些词汇排除
                if phrase in ['phrase', 'name']:
                    continue
                # if not has_uppercase(phrase) and not has_number(phrase):
                #     continue
                ner_query_phrases.add(phrase)
            batch_query_phrases.append(ner_query_phrases)

        return batch_query_phrases

    def graph_retrieve(self, batch_queries):
        # 第一步，对每一个query进行分解
        logger.info("Step1: LLM Generate Subqueries...")
        logger.info(f"Total: {self.config.debug.enable_query_decompose}, Subqueris:{self.config.debug.enable_subquery}, rewrite: {self.config.debug.enable_rewrite_query}, cloze: {self.config.debug.enable_cloze}")
        start = time.time()
        batch_subclozes, batch_rewritten_query, batch_cloze_phrases = self.query_optimizer.batch_decompose(batch_queries,
                                                                                                           enable_cache=True,
                                                                                                           enable_subquery=self.config.debug.enable_subquery,
                                                                                                           enable_query_decompose=self.config.debug.enable_query_decompose,
                                                                                                           enable_rewrite_query=self.config.debug.enable_rewrite_query,
                                                                                                           enable_cloze=self.config.debug.enable_cloze)


        step1 = time.time()
        # 第2步：使用spacy抽取出查询相关的命名实体
        logger.info("Step2: Ner Query...")
        batch_query_phrases = self.batch_ner_query_phrases(batch_queries, batch_cloze_phrases)
        step2 = time.time()

        # 第3步：使用base_retriever检索问题相关文档
        logger.info("Step3: Retrieve seed Passages...")
        batch_document_with_scores = self.batch_retrieve(batch_queries, top_k=self.config.rag.search_doc_num,
                                                         search_mode='query_to_doc')  # top_k为浮点数即1.0=100%

        step3 = time.time()

        # 开始初始化query
        batch_query_solutions = []
        for query, subclozes, rewritten_query, query_phrases, sequence_document_with_scores in zip(batch_queries,batch_subclozes,batch_rewritten_query,batch_query_phrases,batch_document_with_scores):
            ranked_docs = []
            ranked_doc_keys = []
            ranked_doc_scores = []
            for text, key, score in sequence_document_with_scores:
                ranked_docs.append(text)
                ranked_doc_keys.append(key)
                ranked_doc_scores.append(score)

            batch_query_solutions.append(QuerySolution(query=query,
                                                       subqueries=subclozes,
                                                       rewritten_query=rewritten_query,
                                                       query_phrases=query_phrases,
                                                       ranked_docs=ranked_docs,
                                                       ranked_doc_keys=ranked_doc_keys,
                                                       ranked_doc_scores=np.array(ranked_doc_scores),
                                                       history_iter_pre_ppr_reranked_docs=[],
                                                       history_iter_prr_reranked_docs=[],
                                                       evidence_note={},
                                                       evidence_update_status=False,
                                                       filter_responses=[],
                                                       fully_supported_sentence_nodes=set(),
                                                       pre_ppr_eval_recall=[],
                                                       eval_recall=[]))


        logger.info("Step4: Online Iterative Retrieval...")
        if self.config.rag.search_epochs <= 0:
            # 不进行filer，即base_retriever检索
            for solution in batch_query_solutions:
                solution.reranked_docs = solution.ranked_docs
                solution.reranked_doc_keys = solution.ranked_doc_keys
                solution.reranked_doc_scores = solution.ranked_doc_scores
                solution.history_iter_pre_ppr_reranked_docs.append(solution.reranked_doc_keys)     # 记录每一次查询增强的结
                solution.history_iter_prr_reranked_docs.append(solution.reranked_doc_keys)
        else:
            for i in range(self.config.rag.search_epochs):
                self.shallow_memory.run(batch_query_solutions)

        step4 = time.time()
        # 开始迭代检索了
        logger.info(f"""Batch Time Cost:
                    - Total Time: {step4 - start:.3f}s
                    - Step1: LLM Generate Clozes: {step1 - start:.3f}s
                    - Step2: Ner Query: {step2 - step2:.3f}s
                    - Step3: Base Retriver Search Documents: {step3 - step1:.3f}s
                    - Step4: Graph-Based ReRanker Iter Search: {step4 - step3:.3f}s
                    """)

        return batch_query_solutions

    def graph_qa(self, batch_query_solutions):
        all_qa_messages = []
        for query_solution in batch_query_solutions:
            retrieved_documents = query_solution.history_iter_prr_reranked_docs[-1][:self.config.rag.qa_doc_num]
            prompt_user = ''
            for passage in retrieved_documents:
                prompt_user += f'Wikipedia Title: {passage}\n\n'

            # knowledge = []
            # for node_id, item in query_solution.evidence_note.items():
            #     sentence = item['content']
            #     if sentence in prompt_user:
            #         continue
            #     keywords = ';'.join(item['keywords'])
            #     score = item['score']
            #     knowledge.append(f'- Knowledge {len(knowledge)}: (Relevant Keywords: {keywords}) {sentence}')
            # knowledge = '\n\n'.join(knowledge)
            # prompt_user += knowledge
            prompt_user += 'Question: ' + query_solution.query + '\nThought: '
            messages = self.template_manager.render('rag_qa', prompt_user=prompt_user)
            # print(">>>>>>>>>>>>>>>>>>>>Messages>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # print(messages)
            all_qa_messages.append(messages)

        qa_responses = self.base_llm.batch_infer(all_qa_messages, max_completion_tokens=1024, desc='QA Reading', enable_tqdm=True)
        for query_solution, qa_response in zip(batch_query_solutions, qa_responses):
            response_content = qa_response['response']
            # print("??????????????+++++++++++++++++++++++++++++++++++++++")
            # print(response_content)
            # print("+++++++++++++++++++++++++++++++++++++++")
            try:
                # pred_ans = re.findall(r'box\{(.*)\}', response_content)[0]
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                print(response_content)
                pred_ans = response_content
                print("+++++++++++++++++++++++++++++++++++++++")
            # print("===================================")
            # print("Gold: ", query_solution.gold_answer)
            # print("Pred Answer: ", pred_ans)
            query_solution.response = pred_ans

        return batch_query_solutions