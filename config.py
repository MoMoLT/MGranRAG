class HotpotQAConfig:
    class dataset:
        dataset_name = 'sample100'  # 'musique'#'hotpot'#2wikimultihopqa
        dataset_path = f'./datasets/{dataset_name}/{dataset_name}.json'
        corpus_path = f'./datasets/{dataset_name}/{dataset_name}_corpus.json'

        output_root = f'./outputs/{dataset_name}'
        log_path = f'{output_root}/logs'

    class rag:
        max_embedding_length = 2048  # 2048
        embedding_path = '/root/autodl-tmp/NV-Embed-v2'  # contriever-msmarco'#all-MiniLM-L6-v2'  # 基础embedding模型
        # embedding_path = '/root/autodl-fs/contriever-msmarco'
        model_mode = 'instruction'#'instruction'          # 如果需要用到NVEmbedV2模型，使用instruction模式
        batch_size = 64
        ppr_structure = 'd2s2e'             # 图的结构： s2e:sentence-entity    d2s2e: doc-sentence-entity
        max_workers = 32                    # 线程数量
        search_doc_num = 200                # 精炼passage序列的长度
        final_search_doc_num = 10           # 最终检索器检索返回的文档数量，要估算recall的文档
        filter_doc_num = 10                 # 一个问题最多处理的topk个句子
        qa_doc_num = 5                      # 最终只取top_k个文档做问答
        damping = 0.5
        search_epochs = 1

    class debug:
        # enable_debug_recall = True  # 用于测试 Recall
        # enable_debug_gold = False  # 用于调试，使用gold文档调试
        enable_epoch_filter_single_doc = False  # 在filter时一次是否只处理一个文档

        # 消融实验配置
        enable_query_decompose = True  # 开启查询分解和重写
        enable_rewrite_query = False     # 是否开启重写
        enable_subquery = True          # 开启subquery
        enable_cloze = False          # 是否转为cloze

        enable_query_augment = True  # 开启查询增强
        enable_graph_reranker = True  # 开启基于图的reranker，即开启PPR

        enable_weight_doc_node = True  # 开启文档节点赋权
        enable_weight_sentence_node = True
        enable_weight_phrase_node = True
        enable_weight_query_phrase_node = True

    # 0.5, 1, 0.1, 1, 2, 2.5, 4, 1, 0.5 (92)
    class shallow_memory:
        match_node_num = 10
        match_threshold = 0.4
        match_same_threshold = 0.9
        # # 基础缩放因子
        document_ratio = 4  # [0,1]
        sentence_ratio = 1.5  # [3,0]  # 句子基础分数
        phrase_ratio = 8  # *sentence_ratio * node_idfs

        entity_phrase_factor = 1.5
        concept_phrase_factor = 1.5  # 1 / 61

        query_phrase_factor = 1
        sentence_phrase_factor = 1

        filtered_phrase_factor = 2
        unfiltered_phrase_factor = 1/610  # 1 / 61

    class llm:
        api_key = 'empty'
        base_url = 'http://127.0.0.1:8000/v1'
        model = 'Qwen3-8B'  # 'llama3.3:70b'  # 'Meta-Llama-3-8B-Instruct'#
        model_path = '/root/autodl-tmp/Qwen3-8B'
        enable_online = True

    class extractor:
        spacy_model = 'en_core_web_sm'  # spacy加载的模型

    class tqdm:
        enable_ner_tqdm = True  # Extracotr启动NER进度条
        enable_add_tqdm = True  # Retriever启动新增向量进度条
        enable_search_tqdm = False  # Retriever启动向量检索进度条
        enable_filter_tqdm = True  # Filter启动进度条

    class templates:
        root_package = 'prompts'
        filter_facts = f"{root_package}.filter_facts"
        subquery = f"{root_package}.subquery"
        rag_qa = f"{root_package}.rag_qa"
        augment_query = f"{root_package}.augment_query"
