from utils.file_manager import load_json, save_json
from utils.text_manager import generate_key

def get_hotpotqa_samples(dataset_path, corpus_path):
    dataset = load_json(dataset_path)
    corpus = load_json(corpus_path)
    title2content = {}
    for item in corpus:
        title = item['title']
        text = item['text']
        title2content[title] = f"{title}\n\n{text}"

    all_queries = []
    all_gold_docs = []
    all_truths = []
    for data in dataset:
        gold_titles = set()
        for title, idx in data['supporting_facts']:
            gold_titles.add(title)
        gold_docs = []
        for title in gold_titles:
            gold_docs.append(title2content[title])
        gold_docs = sorted(gold_docs)
        data['gold_docs'] = gold_docs

        gold_ans = data['answer']
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if 'answer_aliases' in data:
            gold_ans.update(data['answer_aliases'])
        all_truths.append(gold_ans)

        all_queries.append(data['question'])
        all_gold_docs.append(gold_docs)

    return all_queries, all_gold_docs, all_truths

def get_musique_samples(dataset_path, corpus_path):
    dataset = load_json(dataset_path)
    corpus = load_json(corpus_path)
    key2content = {}
    for item in corpus:
        title = item['title']
        text = item['text']
        content = f"{title}\n\n{text}"
        key = generate_key(content)
        key2content[key] = content

    all_queries = []
    all_gold_docs = []
    all_truths = []
    for data in dataset:
        gold_docs = list()
        for item in data['paragraphs']:
            if item['is_supporting']:
                title = item['title']
                text = item['paragraph_text']
                content = f"{title}\n\n{text}"
                key = generate_key(content)
                gold_docs.append(key2content[key])
        gold_docs = sorted(gold_docs)
        data['gold_docs'] = gold_docs
        gold_ans = data['answer']
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if 'answer_aliases' in data:
            gold_ans.update(data['answer_aliases'])
        all_truths.append(gold_ans)
        all_queries.append(data['question'])
        all_gold_docs.append(gold_docs)
    return all_queries, all_gold_docs, all_truths

def get_popqa_samples(dataset_path, corpus_path):
    dataset = load_json(dataset_path)
    corpus = load_json(corpus_path)
    key2content = {}
    for item in corpus:
        title = item['title']
        text = item['text']
        content = f"{title}\n\n{text}"
        key = generate_key(content)
        key2content[key] = content

    all_queries = []
    all_gold_docs = []
    all_truths = []
    for data in dataset:
        gold_docs = list()
        for item in data['paragraphs']:
            if item['is_supporting']:
                title = item['title']
                text = item['text']
                content = f"{title}\n\n{text}"
                key = generate_key(content)
                gold_docs.append(key2content[key])
        gold_docs = sorted(gold_docs)
        data['gold_docs'] = gold_docs

        gold_ans = set(
            [data['obj']] + [data['possible_answers']] + [data['o_wiki_title']] + [data['o_aliases']])
        gold_ans = set(gold_ans)
        if 'answer_aliases' in data:
            gold_ans.update(data['answer_aliases'])

        all_truths.append(gold_ans)
        all_queries.append(data['question'])
        all_gold_docs.append(gold_docs)
    return all_queries, all_gold_docs, all_truths

def get_nq_samples(dataset_path, corpus_path):
    dataset = load_json(dataset_path)
    corpus = load_json(corpus_path)
    key2content = {}
    for item in corpus:
        title = item['title']
        text = item['text']
        content = f"{title}\n\n{text}"
        key = generate_key(content)
        key2content[key] = content

    all_queries = []
    all_gold_docs = []
    all_truths = []
    for data in dataset:
        gold_docs = list()
        for item in data['contexts']:
            if item['is_supporting']:
                title = item['title']
                text = item['text']
                content = f"{title}\n\n{text}"
                key = generate_key(content)
                gold_docs.append(key2content[key])
        gold_docs = sorted(gold_docs)
        data['gold_docs'] = gold_docs
        gold_ans = data['reference']
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if 'answer_aliases' in data:
            gold_ans.update(data['answer_aliases'])
        all_truths.append(gold_ans)
        all_queries.append(data['question'])
        all_gold_docs.append(gold_docs)
    return all_queries, all_gold_docs, all_truths
