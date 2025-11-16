from blingfire import text_to_sentences_and_offsets
from utils.text_manager import has_uppercase
def get_sentences(text):
    sentences = []
    offsets = text_to_sentences_and_offsets(text)[1]
    for ofs in offsets:
        sentences.append(text[ofs[0]:ofs[1]])
    return sentences
def extract_keywords2(spacy_doc):
    first_stack_word = []
    first_stack_shape = []
    first_stack_pos = []
    first_stack_pos_list = []
    first_stack_word_list = []

    # 第一步是将 - 的词合并和's的词合并
    is_right = False  # 即是否将右边的单词合并
    for token in spacy_doc:
        if token.shape_ in ["'x", "-"] and token.pos_ in ['PUNCT', 'PART']:
            if token.shape_ == '-':
                is_right = True
            if len(first_stack_word):
                # 将当前'-'或者's，合并到前一个单词上，并视为PROPN
                first_stack_word[-1] += token.text
                first_stack_pos[-1] = 'NP'
                first_stack_pos_list[-1].append(token.pos_)
                first_stack_word_list[-1].append(token.text)
        elif is_right:
            is_right = False
            if len(first_stack_word):
                first_stack_word[-1] += token.text
                first_stack_pos[-1] = 'NP'
                first_stack_pos_list[-1].append(token.pos_)
                first_stack_word_list[-1].append(token.text)
        else:
            first_stack_word.append(token.text)
            first_stack_shape.append(token.shape_)
            first_stack_pos.append(token.pos_)
            first_stack_pos_list.append([token.pos_])
            first_stack_word_list.append([token.text])

    # 第二步，即将所有大写词合并
    second_stack_word = []
    second_stack_shape = []
    second_stack_pos = []
    second_stack_pos_list = []
    second_stack_word_list = []
    for cur in range(len(first_stack_word)):
        cur_word = first_stack_word[cur]
        cur_shape = first_stack_shape[cur]
        cur_pos = first_stack_pos[cur]
        cur_pos_list = first_stack_pos_list[cur]
        cur_word_list = first_stack_word_list[cur]

        # 第一步，判断是否为大写词，如果是大写词，那么判断前面的是否存在大写词，如果存在则合并
        if 'X' in cur_shape:
            if len(second_stack_word) == 0:
                # 是首字母
                second_stack_word.append(cur_word)
                second_stack_shape.append(cur_shape + 'Start')
                second_stack_pos.append(cur_pos)
                second_stack_pos_list.append(cur_pos_list)
                second_stack_word_list.append(cur_word_list)

            else:
                # 不是首字母
                # 判断前面单词是否存在大写词
                if 'X' in second_stack_shape[-1]:
                    # 合并
                    second_stack_word[-1] += ' ' + cur_word
                    second_stack_shape[-1] += 'X'
                    if second_stack_pos[-1] != 'PROPN':
                        second_stack_pos[-1] = 'NX'
                    second_stack_pos_list[-1].extend(cur_pos_list)
                    second_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    second_stack_word.append(cur_word)
                    second_stack_shape.append(cur_shape)
                    second_stack_pos.append(cur_pos)
                    second_stack_pos_list.append(cur_pos_list)
                    second_stack_word_list.append(cur_word_list)
        elif cur_pos in ['ADP', 'CCONJ', 'DET', 'PART']:
            # 判断前面是否为大写词
            if len(second_stack_word) == 0:
                # 是首字母
                second_stack_word.append(cur_word)
                second_stack_shape.append(cur_shape + 'Start')
                second_stack_pos.append(cur_pos)
                second_stack_pos_list.append(cur_pos_list)
                second_stack_word_list.append(cur_word_list)
            else:
                if 'X' in second_stack_shape[-1]:
                    # 合并
                    second_stack_word[-1] += ' ' + cur_word
                    second_stack_shape[-1] += 'X'
                    if second_stack_pos[-1] != 'PROPN':
                        second_stack_pos[-1] = 'NX'
                    second_stack_pos_list[-1].extend(cur_pos_list)
                    second_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    second_stack_word.append(cur_word)
                    second_stack_shape.append(cur_shape)
                    second_stack_pos.append(cur_pos)
                    second_stack_pos_list.append(cur_pos_list)
                    second_stack_word_list.append(cur_word_list)
        else:
            second_stack_word.append(cur_word)
            second_stack_shape.append(cur_shape)
            second_stack_pos.append(cur_pos)
            second_stack_pos_list.append(cur_pos_list)
            second_stack_word_list.append(cur_word_list)

    final_stack_word = []
    final_stack_shape = []
    final_stack_pos = []
    final_stack_pos_list = []
    final_stack_word_list = []
    # 第三步，将连续的名词、综合词、数字合并
    for cur in range(len(second_stack_word)):
        cur_word = second_stack_word[cur]
        cur_shape = second_stack_shape[cur]
        cur_pos = second_stack_pos[cur]
        cur_pos_list = second_stack_pos_list[cur]
        cur_word_list = second_stack_word_list[cur]

        # 第一步，判断是否为处理后的词，或者名词、NUM
        if cur_pos in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP']:
            if len(final_stack_word) == 0:
                # 是首字母
                final_stack_word.append(cur_word)
                final_stack_shape.append(cur_shape)
                final_stack_pos.append(cur_pos)
                final_stack_pos_list.append(cur_pos_list)
                final_stack_word_list.append(cur_word_list)
            else:
                # 不是首字母
                # 判断前面单词是否为满足条件的词
                if final_stack_pos[-1] in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP', 'NNN']:
                    # 合并
                    final_stack_word[-1] += ' ' + cur_word
                    final_stack_shape[-1] += 'X'
                    if final_stack_pos[-1] != 'PROPN':
                        final_stack_pos[-1] = 'NNN'
                    final_stack_pos_list[-1].extend(cur_pos_list)
                    final_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    final_stack_word.append(cur_word)
                    final_stack_shape.append(cur_shape)
                    final_stack_pos.append(cur_pos)
                    final_stack_pos_list.append(cur_pos_list)
                    final_stack_word_list.append(cur_word_list)
        else:
            final_stack_word.append(cur_word)
            final_stack_shape.append(cur_shape)
            final_stack_pos.append(cur_pos)
            final_stack_pos_list.append(cur_pos_list)
            final_stack_word_list.append(cur_word_list)
    keywords = set()
    # 最终取出合并的词汇
    for cur in range(len(final_stack_word)):
        cur_word = final_stack_word[cur]
        cur_shape = final_stack_shape[cur]
        cur_pos = final_stack_pos[cur]
        cur_pos_list = final_stack_pos_list[cur]
        cur_word_list = final_stack_word_list[cur]

        if cur_pos in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP', 'NNN']:
            keywords.add(cur_word)
            # 同时把合并词中的所有PROPN都取出来，原因是Bob and Lucy会视为合并词，有时电影叫这个名，没办法，有时候又可以合并
            if 'PROPN' in cur_pos_list:
                cur_keywords = set()
                for pidx, pos in enumerate(cur_pos_list):
                    if pos == 'PROPN':
                        cur_keywords.add(cur_word_list[pidx])
                    else:
                        keywords.add(' '.join(list(cur_keywords)))
                        cur_keywords = set()

    return keywords

def extract_keywords3(spacy_doc):
    first_stack_word = []
    first_stack_shape = []
    first_stack_pos = []
    first_stack_pos_list = []
    first_stack_word_list = []

    # 第一步是将 - 的词合并和's的词合并
    is_right = False  # 即是否将右边的单词合并
    for token in spacy_doc:
        if token.shape_ in ["'x", "-"] and token.pos_ in ['PUNCT', 'PART']:
            if token.shape_ == '-':
                is_right = True
            if len(first_stack_word):
                # 将当前'-'或者's，合并到前一个单词上，并视为PROPN
                first_stack_word[-1] += token.text
                first_stack_pos[-1] = 'NP'
                first_stack_pos_list[-1].append(token.pos_)
                first_stack_word_list[-1].append(token.text)
        elif is_right:
            is_right = False
            if len(first_stack_word):
                first_stack_word[-1] += token.text
                first_stack_pos[-1] = 'NP'
                first_stack_pos_list[-1].append(token.pos_)
                first_stack_word_list[-1].append(token.text)
        else:
            first_stack_word.append(token.text)
            first_stack_shape.append(token.shape_)
            first_stack_pos.append(token.pos_)
            first_stack_pos_list.append([token.pos_])
            first_stack_word_list.append([token.text])
    # print('first: ', first_stack_word, first_stack_shape, first_stack_pos, first_stack_pos_list)

    # 第二步，即将所有大写词合并
    second_stack_word = []
    second_stack_shape = []
    second_stack_pos = []
    second_stack_pos_list = []
    second_stack_word_list = []
    for cur in range(len(first_stack_word)):
        cur_word = first_stack_word[cur]
        cur_shape = first_stack_shape[cur]
        cur_pos = first_stack_pos[cur]
        cur_pos_list = first_stack_pos_list[cur]
        cur_word_list = first_stack_word_list[cur]

        # 第一步，判断是否为大写词，如果是大写词，那么判断前面的是否存在大写词，如果存在则合并
        if 'X' in cur_shape:
            if len(second_stack_word) == 0:
                # 是首字母
                second_stack_word.append(cur_word)
                second_stack_shape.append(cur_shape + 'Start')
                second_stack_pos.append(cur_pos)
                second_stack_pos_list.append(cur_pos_list)
                second_stack_word_list.append(cur_word_list)

            else:
                # 不是首字母
                # 判断前面单词是否存在大写词
                if 'X' in second_stack_shape[-1]:
                    # 合并
                    second_stack_word[-1] += ' ' + cur_word
                    second_stack_shape[-1] += 'X'
                    if second_stack_pos[-1] != 'PROPN':
                        second_stack_pos[-1] = 'NX'
                    second_stack_pos_list[-1].extend(cur_pos_list)
                    second_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    second_stack_word.append(cur_word)
                    second_stack_shape.append(cur_shape)
                    second_stack_pos.append(cur_pos)
                    second_stack_pos_list.append(cur_pos_list)
                    second_stack_word_list.append(cur_word_list)
        elif cur_pos in ['ADP', 'CCONJ', 'DET', 'PART']:
            # 判断前面是否为大写词
            if len(second_stack_word) == 0:
                # 是首字母
                second_stack_word.append(cur_word)
                second_stack_shape.append(cur_shape + 'Start')
                second_stack_pos.append(cur_pos)
                second_stack_pos_list.append(cur_pos_list)
                second_stack_word_list.append(cur_word_list)
            else:
                if 'X' in second_stack_shape[-1]:
                    # 合并
                    second_stack_word[-1] += ' ' + cur_word
                    second_stack_shape[-1] += 'X'
                    if second_stack_pos[-1] != 'PROPN':
                        second_stack_pos[-1] = 'NX'
                    second_stack_pos_list[-1].extend(cur_pos_list)
                    second_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    second_stack_word.append(cur_word)
                    second_stack_shape.append(cur_shape)
                    second_stack_pos.append(cur_pos)
                    second_stack_pos_list.append(cur_pos_list)
                    second_stack_word_list.append(cur_word_list)
        else:
            second_stack_word.append(cur_word)
            second_stack_shape.append(cur_shape)
            second_stack_pos.append(cur_pos)
            second_stack_pos_list.append(cur_pos_list)
            second_stack_word_list.append(cur_word_list)

    # print('second: ', second_stack_word, second_stack_shape, second_stack_pos, second_stack_pos_list)
    final_stack_word = []
    final_stack_shape = []
    final_stack_pos = []
    final_stack_pos_list = []
    final_stack_word_list = []
    # 第三步，将连续的名词、综合词、数字合并
    for cur in range(len(second_stack_word)):
        cur_word = second_stack_word[cur]
        cur_shape = second_stack_shape[cur]
        cur_pos = second_stack_pos[cur]
        cur_pos_list = second_stack_pos_list[cur]
        cur_word_list = second_stack_word_list[cur]

        # 第一步，判断是否为处理后的词，或者名词、NUM
        if cur_pos in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP']:
            if len(final_stack_word) == 0:
                # 是首字母
                final_stack_word.append(cur_word)
                final_stack_shape.append(cur_shape)
                final_stack_pos.append(cur_pos)
                final_stack_pos_list.append(cur_pos_list)
                final_stack_word_list.append(cur_word_list)
            else:
                # 不是首字母
                # 判断前面单词是否为满足条件的词
                if final_stack_pos[-1] in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP', 'NNN']:
                    # 合并
                    final_stack_word[-1] += ' ' + cur_word
                    final_stack_shape[-1] += 'X'
                    if final_stack_pos[-1] != 'PROPN':
                        final_stack_pos[-1] = 'NNN'
                    final_stack_pos_list[-1].extend(cur_pos_list)
                    final_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    final_stack_word.append(cur_word)
                    final_stack_shape.append(cur_shape)
                    final_stack_pos.append(cur_pos)
                    final_stack_pos_list.append(cur_pos_list)
                    final_stack_word_list.append(cur_word_list)
        else:
            final_stack_word.append(cur_word)
            final_stack_shape.append(cur_shape)
            final_stack_pos.append(cur_pos)
            final_stack_pos_list.append(cur_pos_list)
            final_stack_word_list.append(cur_word_list)
    # print('final: ', final_stack_word, final_stack_shape, final_stack_pos, final_stack_pos_list)
    keywords = set()
    # 最终取出合并的词汇
    for cur in range(len(final_stack_word)):
        cur_word = final_stack_word[cur]
        cur_shape = final_stack_shape[cur]
        cur_pos = final_stack_pos[cur]
        cur_pos_list = final_stack_pos_list[cur]
        cur_word_list = final_stack_word_list[cur]

        if cur_pos in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP', 'NNN']:
            # 首先判断cur_word最后一个是否为PROPN, 如果是则加入，如果不是则截断
            if cur_pos_list[-1] in ['PROPN', 'NOUN', 'NUM']:
                keywords.add(cur_word)
            else:
                # 进行截断处理
                for i in range(len(cur_pos_list) - 1, 0, -1):
                    if cur_pos_list[i] in ['PROPN', 'NOUN', 'NUM']:
                        break
                keywords.add(' '.join(cur_word_list[:i]))

            # 同时把合并词中的所有PROPN都取出来，原因是Bob and Lucy会视为合并词，有时电影叫这个名，没办法，有时候又可以合并
            if 'PROPN' in cur_pos_list:
                cur_keywords = set()
                for pidx, pos in enumerate(cur_pos_list):
                    if pos == 'PROPN':
                        cur_keywords.add(cur_word_list[pidx])
                    else:
                        if len(cur_keywords):
                            keywords.add(' '.join(list(cur_keywords)))
                        cur_keywords = set()
                if len(cur_keywords):
                    keywords.add(' '.join(list(cur_keywords)))

    return keywords

def extract_keywords(spacy_doc):
    first_stack_word = []
    first_stack_shape = []
    first_stack_pos = []
    first_stack_pos_list = []
    first_stack_word_list = []
    # 第一步是将 - 的词合并和's的词合并
    is_right = False  # 即是否将右边的单词合并
    for token in spacy_doc:
        if token.shape_ in ["'x", "-"] and token.pos_ in ['PUNCT', 'PART']:
            if token.shape_ == '-':
                is_right = True
            if len(first_stack_word):
                # 将当前'-'或者's，合并到前一个单词上，并视为PROPN
                first_stack_word[-1] += token.text
                first_stack_pos[-1] = 'NP'
                first_stack_pos_list[-1].append(token.pos_)
                first_stack_word_list[-1].append(token.text)
        elif is_right:
            is_right = False
            if len(first_stack_word):
                first_stack_word[-1] += token.text
                first_stack_pos[-1] = 'NP'
                first_stack_pos_list[-1].append(token.pos_)
                first_stack_word_list[-1].append(token.text)
        else:
            first_stack_word.append(token.text)
            first_stack_shape.append(token.shape_)
            first_stack_pos.append(token.pos_)
            first_stack_pos_list.append([token.pos_])
            first_stack_word_list.append([token.text])

    # 第二步，即将所有大写词合并
    second_stack_word = []
    second_stack_shape = []
    second_stack_pos = []
    second_stack_pos_list = []
    second_stack_word_list = []
    for cur in range(len(first_stack_word)):
        cur_word = first_stack_word[cur]
        cur_shape = first_stack_shape[cur]
        cur_pos = first_stack_pos[cur]
        cur_pos_list = first_stack_pos_list[cur]
        cur_word_list = first_stack_word_list[cur]

        # 第一步，判断是否为大写词，如果是大写词，那么判断前面的是否存在大写词，如果存在则合并
        if 'X' in cur_shape:
            if len(second_stack_word) == 0:
                # 是首字母
                second_stack_word.append(cur_word)
                second_stack_shape.append(cur_shape + 'Start')
                second_stack_pos.append(cur_pos)
                second_stack_pos_list.append(cur_pos_list)
                second_stack_word_list.append(cur_word_list)

            else:
                # 不是首字母
                # 判断前面单词是否存在大写词
                if 'X' in second_stack_shape[-1]:
                    # 合并
                    second_stack_word[-1] += ' ' + cur_word
                    second_stack_shape[-1] += 'X'
                    if second_stack_pos[-1] != 'PROPN':
                        second_stack_pos[-1] = 'NX'
                    second_stack_pos_list[-1].extend(cur_pos_list)
                    second_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    second_stack_word.append(cur_word)
                    second_stack_shape.append(cur_shape)
                    second_stack_pos.append(cur_pos)
                    second_stack_pos_list.append(cur_pos_list)
                    second_stack_word_list.append(cur_word_list)
        elif cur_pos in ['ADP', 'CCONJ', 'DET', 'PART']:
            # 判断前面是否为大写词
            if len(second_stack_word) == 0:
                # 是首字母
                second_stack_word.append(cur_word)
                second_stack_shape.append(cur_shape + 'Start')
                second_stack_pos.append(cur_pos)
                second_stack_pos_list.append(cur_pos_list)
                second_stack_word_list.append(cur_word_list)
            else:
                if 'X' in second_stack_shape[-1]:
                    # 合并
                    second_stack_word[-1] += ' ' + cur_word
                    second_stack_shape[-1] += 'X'
                    if second_stack_pos[-1] != 'PROPN':
                        second_stack_pos[-1] = 'NX'
                    second_stack_pos_list[-1].extend(cur_pos_list)
                    second_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    second_stack_word.append(cur_word)
                    second_stack_shape.append(cur_shape)
                    second_stack_pos.append(cur_pos)
                    second_stack_pos_list.append(cur_pos_list)
                    second_stack_word_list.append(cur_word_list)
        else:
            second_stack_word.append(cur_word)
            second_stack_shape.append(cur_shape)
            second_stack_pos.append(cur_pos)
            second_stack_pos_list.append(cur_pos_list)
            second_stack_word_list.append(cur_word_list)

    # print('second: ', second_stack_word, second_stack_shape, second_stack_pos, second_stack_pos_list)
    final_stack_word = []
    final_stack_shape = []
    final_stack_pos = []
    final_stack_pos_list = []
    final_stack_word_list = []

    # 第三步，将连续的名词、综合词、数字合并
    for cur in range(len(second_stack_word)):
        cur_word = second_stack_word[cur]
        cur_shape = second_stack_shape[cur]
        cur_pos = second_stack_pos[cur]
        cur_pos_list = second_stack_pos_list[cur]
        cur_word_list = second_stack_word_list[cur]

        # 第一步，判断是否为处理后的词，或者名词、NUM
        if cur_pos in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP']:
            if len(final_stack_word) == 0:
                # 是首字母
                final_stack_word.append(cur_word)
                final_stack_shape.append(cur_shape)
                final_stack_pos.append(cur_pos)
                final_stack_pos_list.append(cur_pos_list)
                final_stack_word_list.append(cur_word_list)
            else:
                # 不是首字母
                # 判断前面单词是否为满足条件的词
                if final_stack_pos[-1] in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP', 'NNN']:
                    # 合并
                    final_stack_word[-1] += ' ' + cur_word
                    final_stack_shape[-1] += 'X'
                    if final_stack_pos[-1] != 'PROPN':
                        final_stack_pos[-1] = 'NNN'
                    final_stack_pos_list[-1].extend(cur_pos_list)
                    final_stack_word_list[-1].extend(cur_word_list)
                else:
                    # 放入最后一个
                    final_stack_word.append(cur_word)
                    final_stack_shape.append(cur_shape)
                    final_stack_pos.append(cur_pos)
                    final_stack_pos_list.append(cur_pos_list)
                    final_stack_word_list.append(cur_word_list)
        else:
            final_stack_word.append(cur_word)
            final_stack_shape.append(cur_shape)
            final_stack_pos.append(cur_pos)
            final_stack_pos_list.append(cur_pos_list)
            final_stack_word_list.append(cur_word_list)
    # print('final: ', final_stack_word, final_stack_shape, final_stack_pos, final_stack_pos_list)
    keywords = set()
    # 最终取出合并的词汇
    for cur in range(len(final_stack_word)):
        cur_word = final_stack_word[cur]
        cur_shape = final_stack_shape[cur]
        cur_pos = final_stack_pos[cur]
        cur_pos_list = final_stack_pos_list[cur]
        cur_word_list = final_stack_word_list[cur]
        if cur_pos in ['PROPN', 'NOUN', 'NUM', 'NX', 'NP', 'NNN']:
            # 首先判断cur_word最后一个是否为PROPN, 如果是则加入，如果不是则截断
            # 如果最后是一个小写词汇，则需要判断
            if not has_uppercase(cur_word_list[-1]) and cur_pos_list[-1] not in ['PROPN', 'NOUN', 'NUM', 'PART']:
                # 进行截断处理
                for i in range(len(cur_pos_list) - 1, 0, -1):
                    if cur_pos_list[i] in ['PROPN', 'NOUN', 'NUM', 'PART'] or has_uppercase(cur_word_list[i]):
                        break
                # 不能简单的拼接
                word = ' '.join(cur_word_list[:i+1])
                word = replace_word(word)
                keywords.add(word)
            else:
                word = replace_word(cur_word)
                keywords.add(word)

            # 同时把合并词中的所有PROPN都取出来，原因是Bob and Lucy会视为合并词，有时电影叫这个名，没办法，有时候又可以合并
            is_pick = False
            for pos in ['PROPN', 'NOUN', 'NUM']:
                if pos in cur_pos_list:
                    is_pick = True
                    break
            if is_pick:
                cur_keywords = []
                for pidx, pos in enumerate(cur_pos_list):
                    if pos == 'CCONJ' and cur_word_list[pidx][0].islower():
                        # 对于and , or需要断开
                        if len(cur_keywords):
                            word = ' '.join(cur_keywords)
                            word = replace_word(word)
                            keywords.add(word)
                        cur_keywords = []
                    else:
                        cur_keywords.append(cur_word_list[pidx])
                if len(cur_keywords):
                    word = ' '.join(cur_keywords)
                    word = replace_word(word)
                    keywords.add(word)

    return keywords

def replace_word(word):
    return word.replace(' - ', '-').replace(" 's", "'s").replace(' -', '-').replace('- ', '-').replace(" 'S", "'S")
def get_ner(spacy_doc):
    entities_dict = {}
    for ent in spacy_doc.ents:
        text = ent.text.strip()
        texts = text.split('\n')
        for t in texts:
            entities_dict[t] = ent.label_
    return entities_dict
def ner_all_keywords(spacy_doc):
    keywords = extract_keywords(spacy_doc)
    ner_dict = get_ner(spacy_doc)
    return set(keywords).union(ner_dict.keys())
