# 分解
import re
class QuestionOptimizer(object):
    def __init__(self, llm, template_manager, enable_tqdm=True):
        self.llm = llm
        self.template_manager = template_manager
        self.enable_tqdm = enable_tqdm

    def strip_split(self, response, enable_subquery=True, enable_rewrite_query=True, enable_cloze=True):
        # 抽取子填空题
        # 接着是抽取template
        if enable_cloze:
            pattern_subcloze = 'template'
            pattern_rewrite = 'COMBINED_TEMPLATE'
        else:
            pattern_subcloze = 'subquery'
            pattern_rewrite = 'REWRITTEN_QUERY'
        subclozes = re.findall(pattern_subcloze+':(.*?)\n', response, re.S)
        all_clozes = set()
        for subcloze in subclozes:
            subcloze = subcloze.strip()
            if len(subcloze) == 0:
                continue
            if subcloze not in all_clozes:
                all_clozes.add(subcloze)
        # 抽取主cloze
        rewritten_query = re.findall(pattern_rewrite+'\{(.*?)\}', response)
        if len(rewritten_query) == 0:
            rewritten_query = re.findall(pattern_rewrite+'(.*)', response)
        if len(rewritten_query) and enable_rewrite_query:
            rewritten_query = rewritten_query[0].strip()
            if len(rewritten_query) == 0:
                pass
            else:
                if 'OVER OUTPUT' in rewritten_query:
                    rewritten_query = rewritten_query.replace('OVER OUTPUT', '')
                if '}' == rewritten_query[-1]:
                    rewritten_query = rewritten_query[:-1]
                if '{' == rewritten_query[0]:
                    rewritten_query = rewritten_query[1:]
        else:
            rewritten_query = None

        all_entities = set()
        all_clozes = list(all_clozes - {rewritten_query})

        tmp_list = all_clozes if rewritten_query is None else all_clozes+[rewritten_query]
        for cloze in tmp_list:
            entities = re.findall('\[(.*?)\]', cloze)
            for e in entities:
                e = e.strip()
                if e == '?' or '&' in e:
                    continue
                all_entities.add(e)
        if not enable_subquery:
            all_clozes = []
        return sorted(all_clozes), rewritten_query, all_entities

    # 问题分解&重写
    def batch_decompose(self, batch_queries, enable_cache=True, enable_subquery=True, enable_query_decompose=True, enable_rewrite_query=True, enable_cloze=True):
        if enable_query_decompose:
            batch_messages = []
            for query in batch_queries:
                message = self.template_manager.render('subquery', question=query)
                batch_messages.append(message)
            sequence_split_response = self.llm.batch_infer(batch_messages, enable_cache=enable_cache, max_completion_tokens=1024, desc='LLM Query Decomposer', enable_tqdm=self.enable_tqdm)

            batch_subclozes = []
            batch_rewritten_query = []
            batch_cloze_entities = []
            for query, split_response in zip(batch_queries, sequence_split_response):
                response_str = split_response['response']
                finish_reason = split_response['finish_reason']
                if finish_reason != 'stop':
                    # print("++++++++++++++++++++++FINISH LLM Decomposer++++++++++++++++++++")
                    # print(finish_reason)
                    # print(response_str.strip())
                    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    pass
                sub_clozes, rewritten_query, sub_clozes_entities = self.strip_split(response_str, enable_subquery, enable_rewrite_query, enable_cloze)
                batch_subclozes.append(sub_clozes)
                batch_rewritten_query.append(rewritten_query)
                batch_cloze_entities.append(sub_clozes_entities)

                # print("Query: ", query)
                # print("SubQuery: ", sub_clozes)
                # print("Requery: ", rewritten_query)
                # print("---------------------------------------")

        else:
            data_size = len(batch_queries)
            batch_subclozes = [[] for _ in range(data_size)]
            batch_rewritten_query = [None for _ in range(data_size)]
            batch_cloze_entities = [[] for _ in range(data_size)]

        return batch_subclozes, batch_rewritten_query, batch_cloze_entities

    def batch_augment(self, batch_query_solutions, enable_cache=True):
        batch_queries = []
        for solution in batch_query_solutions:
            query = solution.query
            evidence_note = solution.evidence_note

            supported_evidences = set()
            for node_id, item in evidence_note.items():
                content = item['content']
                keywords = ';'.join(item['keywords'])
                score = item['score']
                # relevant_clozes = item['relevant_clozes']
                if node_id in solution.fully_supported_sentence_nodes:
                    supported_evidences.add((node_id, content, keywords, score))
                # for cloze_id, supported_level in relevant_clozes:
                #     if 'T' == supported_level:
                #         supported_evidences.append((node_id, content, keywords, score))
                #         break
            supported_evidences = sorted(supported_evidences, key=lambda x: x[3], reverse=True)
            knowledge = []
            cloze_num = len(solution.subqueries)
            new_evidence_cnt = 0
            # 首先取出最佳完全证据
            for node, sentence, keywords, score in supported_evidences[:cloze_num]:
                knowledge.append(f'- Knowledge {len(knowledge)}: (Relevant Keywords: {keywords}) {sentence}')
                if node in solution.cur_iter_sentence_nodes:
                    new_evidence_cnt += 1

            # 然后取出本轮加入的最新完全证据
            if new_evidence_cnt < len(solution.cur_iter_sentence_nodes):
                for node, sentence, keywords, score in supported_evidences[cloze_num:]:
                    if node in solution.cur_iter_sentence_nodes:
                        knowledge.append(f'- Knowledge {len(knowledge)}: (Relevant Keywords: {keywords}) {sentence}')
                        new_evidence_cnt += 1
                    if new_evidence_cnt >= len(solution.cur_iter_sentence_nodes):
                        break

            knowledge = '\n'.join(knowledge)

            augment_query = f'{knowledge}\n<SEP>Query: {solution.query}'
            if len(solution.evidence_note) == 0:
                solution.augment_query = solution.query
            else:
                solution.augment_query = augment_query

            batch_queries.append(solution.augment_query)

        return batch_queries





