from utils.schema_manager import FactList, ClueList, InferList
import json
from utils.log_manager import get_logger
import difflib
import re
logger = get_logger(__name__)


class SentenceFilter:
    def __init__(self, llm, tokenizer, template_manager, enable_tqdm=True, enable_epoch_filter_single_doc=True):
        self.llm = llm
        self.template_manager = template_manager
        self.tokenizer = tokenizer

        self.enable_tqdm = enable_tqdm
        self.fact_json_schema = FactList.model_json_schema()
        self.clue_json_schema = ClueList.model_json_schema()
        self.infer_json_schema = InferList.model_json_schema()

        # self.max_input_tokens = self.llm.get_input_max_tokens()
        self.max_tokens = self.llm.get_model_max_tokens()
        self.max_output_tokens = 2048
        self.clue_overlap = 200
        template = self.template_manager.render('filter_facts', clozes='question',
                                                facts='facts',
                                                document='document')
        template = json.dumps(template)
        self.template_token_len, _ = self.get_token_length_and_ids(template)
        self.template_margin_token = 1200

        self.relevance_level_mapping = {'Directly Relevant': 2, 'Contextual Support': 1, 'Potential Bridge Fact': 0}
        self.enable_epoch_filter_single_doc = enable_epoch_filter_single_doc

    def get_token_length_and_ids(self, text):
        token_ids = self.tokenizer.encode(text)
        return len(token_ids), token_ids

    def strip_filter(self, query_mapping, mapping, mapping_fact_ids, mapping_fact_texts, response):
        # res = re.findall('\[sentence (\d*)@(\d*) BEGIN\](.*?)reason:', response, re.S)
        res = re.findall('\[sentence (\d*)@(\d*)\](.*?)reason:', response, re.S)
        evidences = {}  #
        # 为了防止重复fact_id，因为复读现象
        processed_fact_ids = set()
        for line in res:
            doc_id, sentence_id, context = line
            content_res = re.findall(
                'explicit_content:(.*?)supplementary_facts:(.*?)relevant_questions:(.*?)keywords:(.*)',
                context, re.S)
            # print(">>>>>>>>>>>>>>>>>")
            # print(context)
            # print("------------------------------")
            # print(content_res)
            # content_res = re.findall(
            #     'content:(.*?)explicit_content:(.*?)supplementary_facts:(.*?)relevant_clozes:(.*?)keywords:(.*)',
            #     context, re.S)
            if len(content_res) == 0:
                continue
            fact_id = f'{doc_id}@{sentence_id}'
            if fact_id in processed_fact_ids:
                continue
            processed_fact_ids.add(fact_id)
            # content, explicit_content, supplementary_facts, relevant_clozes, keywords_str = content_res[0]
            explicit_content, supplementary_facts, relevant_clozes, keywords_str = content_res[0]
            # relevant_supported_list = re.findall("#(\d*?)#\[([TP])\]", relevant_clozes, re.S)
            relevant_supported_list = re.findall("Q(\d*?)\[([TP])\]", relevant_clozes, re.S)
            supported_scores = {'query': 0, 'rewritten': 0, 'subquery': 0}
            subquery_num = 0
            for query_idx, level in relevant_supported_list:
                query_idx = int(query_idx)
                level_factor = 1 if level == 'T' else 0.5
                query_type = query_mapping.get(query_idx, None)
                if query_type is None:
                    continue
                supported_scores[query_type] += level_factor
                if query_type == 'subquery':
                    subquery_num += 1

            subquery_num = max(subquery_num, 1)

            score = supported_scores['query'] + supported_scores['rewritten'] + supported_scores['subquery']/subquery_num
            if score == 0 or (supported_scores['query']<1 and supported_scores['rewritten'] + supported_scores['subquery'] < 0.5):
                continue
            keywords = set()
            for k in keywords_str.split(';'):
                k = k.strip()
                if len(k):
                    keywords.add(k)
            # keywords = list(keywords)

            if fact_id not in mapping:
                # suggestions = difflib.get_close_matches(content.strip(), mapping_fact_texts)
                suggestions = difflib.get_close_matches(explicit_content.strip(), mapping_fact_texts)
                if len(suggestions) == 0:
                    # print("query: ", query)
                    # print("ERROR ID: ", fact_id)
                    # print("mapping: ", list(mapping.keys()))
                    # print(f"<Fact ID ERROR>!!!! {response}")
                    continue
                index = mapping_fact_texts.index(suggestions[0])
                fact_id = mapping_fact_ids[index]
                node, sentence = mapping[fact_id]
            else:
                node, sentence = mapping[fact_id]
            evidences[node] = {'id': node, 'content': explicit_content.strip(), 'org_content': sentence,
                               'keywords': keywords, 'relevant_clozes': set(relevant_supported_list), 'score': score}

        return evidences
    def batch_filter(self, batch_query_solutions):
        # 首先搜集所有的fact
        all_query_info = {}
        sequence_query = []
        sequence_filter_message = []
        sequence_query_mapping = []
        sequence_filter_mapping = []
        filter_max_token_input = self.max_tokens - self.template_token_len - self.template_margin_token - self.max_output_tokens  #self.max_input_tokens
        for solution in batch_query_solutions:
            # if solution.prediction_status == True:
            #     continue
            query = solution.query
            all_query_info[query] = solution
            subqueries = solution.subqueries
            rewritten_query = solution.rewritten_query
            solution.cur_iter_sentence_nodes = set()
            mapping_query_type = {}
            input_questions = []
            cidx = 10
            # 重要性从小到大
            # input_questions.append(f"[#{cidx}#] {query}")
            input_questions.append(f"Q{cidx}: {query}")
            mapping_query_type[cidx] = 'query'
            cidx += 1
            if rewritten_query is not None:
                # input_questions.append(f"[#{cidx}#] {rewritten_query}")
                input_questions.append(f"Q{cidx}: {rewritten_query}")
                mapping_query_type[cidx] = 'rewritten'
                cidx += 1

            for cloze in subqueries:
                # input_questions.append(f"[#{cidx}#] {cloze}")
                input_questions.append(f"Q{cidx}: {cloze}")
                mapping_query_type[cidx] = 'subquery'
                cidx += 1


            input_questions = '\n'.join(input_questions)
            # print("input_questions: ", input_questions)
            # print(generate_key(input_questions))
            # 提取的evidence: {id:node, content, org_sentence, keywords, score(按照)}
            evidences = solution.evidence_note
            relevant_fact_note = []
            sorted_pairs = [
                (item['id'], item['content'], item['score']) for item in
                sorted(evidences.values(), key=lambda x: x['score'], reverse=True)
            ]

            for fact_idx, (node, sentence, score) in enumerate(sorted_pairs):
                relevant_fact_note.append(f"[fact_{fact_idx}] {sentence}")
            if len(relevant_fact_note) > 0:
                relevant_fact_note = '\n'.join(relevant_fact_note)
            else:
                relevant_fact_note = 'N/A'
            # print("relevant_fact_note: ", relevant_fact_note)
            # print(generate_key(relevant_fact_note))
            # print("++++++++++++++++++++++++++++++++++++++++++++")
            # 句子内容
            unprocessed_fact_texts = solution.unprocessed_fact_texts
            unprocessed_fact_nodes = solution.unprocessed_fact_nodes
            # 待交付给LLM的chunk
            unprocessed_chunk_texts = []
            unprocessed_chunk_nodes = []

            doc_idx = 9  # + cur_idx
            cur_doc_total_tokens = 0
            current_chunk_texts = []
            current_chunk_nodes = []
            # 首先加载文档, 多个文档合并,首先判断是否能够合成一个chunk,根据token长度合并
            for cur_idx, (group_texts, group_nodes) in enumerate(zip(unprocessed_fact_texts, unprocessed_fact_nodes)):
                document_text = '<split>'.join(group_texts)

                token_length, token_ids = self.get_token_length_and_ids(document_text)
                # 处理超长文档
                if token_length > filter_max_token_input:
                    trimmed_token_ids = token_ids[:filter_max_token_input - 5]
                    document_text = self.tokenizer.decode(trimmed_token_ids)
                    token_length = len(trimmed_token_ids)

                sentences = document_text.split('<split>')

                if self.enable_epoch_filter_single_doc:
                    unprocessed_chunk_texts.append(sentences)
                    unprocessed_chunk_nodes.append(group_nodes[:len(sentences)])
                # 接着开始拼接
                else:
                    if token_length + cur_doc_total_tokens > filter_max_token_input:
                        # 即无法拼接, 将之前的chunk加入到队列里
                        unprocessed_chunk_texts.append(current_chunk_texts[:])
                        unprocessed_chunk_nodes.append(current_chunk_nodes[:])
                        current_chunk_texts = []
                        current_chunk_texts.extend(sentences)
                        cur_doc_total_tokens = 0
                    else:
                        current_chunk_texts.extend(sentences)
                        current_chunk_nodes.extend(group_nodes[:len(sentences)])
                        cur_doc_total_tokens += token_length

            if current_chunk_texts:
                unprocessed_chunk_texts.append(current_chunk_texts[:])
                unprocessed_chunk_nodes.append(current_chunk_nodes[:])
            for chunk_texts, chunk_nodes in zip(unprocessed_chunk_texts, unprocessed_chunk_nodes):
                mapping = {}
                doc = []
                for idx, (raw_text, node) in enumerate(zip(chunk_texts, chunk_nodes)):
                    text = raw_text.strip().replace('\n\n', '<SEP>').replace('\n', '<SEP>')
                    sentence_idx = f'{doc_idx}@{idx}'
                    sentence = f'[{sentence_idx}] {text}'
                    doc.append(sentence)
                    mapping[sentence_idx] = (node, raw_text)

                doc = '\n'.join(doc)
                document_text = f'<DOC {doc_idx}>\n{doc}\n</DOC {doc_idx}>'

                message = self.template_manager.render('filter_facts',
                                                       clozes=input_questions,
                                                       facts=relevant_fact_note,
                                                       document=document_text)

                sequence_query.append(query)
                sequence_filter_message.append(message)
                sequence_filter_mapping.append(mapping)
                sequence_query_mapping.append(mapping_query_type)


        # 开始批量处理
        sequence_filter_response = self.llm.batch_infer(sequence_filter_message,
                                                        max_completion_tokens=self.max_output_tokens,
                                                        desc='LLM Filter Facts', enable_tqdm=self.enable_tqdm)
        # extra_body={"guided_json": self.fact_json_schema})

        for query, query_mapping, fact_mapping, fact_response, message in zip(sequence_query, sequence_query_mapping,
                                                                              sequence_filter_mapping,
                                                                              sequence_filter_response,
                                                                              sequence_filter_message):
            response_str = fact_response['response']
            all_query_info[query].filter_responses.append(response_str)
            finish_reason = fact_response['finish_reason']
            if finish_reason != 'stop':
                # print("++++++++++++++++++++++FINISH LLM++++++++++++++++++++")
                # print(finish_reason)
                # print(response_str.strip())
                # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                pass

            mapping_fact_ids = list(fact_mapping.keys())
            mapping_fact_texts = []
            for _id in mapping_fact_ids:
                node, sentence = fact_mapping[_id]
                mapping_fact_texts.append(sentence)

            evidences = self.strip_filter(query_mapping, fact_mapping, mapping_fact_ids, mapping_fact_texts, response_str)
            # fully_supported_sentence_nodes = all_query_info[query].fully_supported_sentence_nodes
            for node, evidence in evidences.items():
                if node not in all_query_info[query].evidence_note:
                    all_query_info[query].evidence_note[node] = evidence
                else:
                    old_evidence = all_query_info[query].evidence_note[node]
                    relevant_clozes = evidence['relevant_clozes'] | old_evidence['relevant_clozes']
                    keywords = evidence['keywords'] | old_evidence['keywords']
                    score = max(evidence['score'], old_evidence['score'])
                    evidence['keywrods'] = keywords
                    evidence['score'] = score
                    evidence['relevant_clozes'] = relevant_clozes
                    all_query_info[query].evidence_note[node] = evidence
                all_query_info[query].evidence_update_status = True

                relevant_clozes = evidence['relevant_clozes']

                for item in relevant_clozes:
                    idx, qtype = item
                    if qtype == 'T':
                        all_query_info[query].fully_supported_sentence_nodes.add(node)
                        all_query_info[query].cur_iter_sentence_nodes.add(node)
                        break


