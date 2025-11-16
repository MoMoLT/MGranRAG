'''
功能简介：在线LLM API调用

加入了sqlite缓存机制，相同参数和提示词的则会调用缓存
'''
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import defaultdict
import os
import httpx
import json
from tenacity import retry, stop_after_attempt, wait_fixed
import functools
from copy import deepcopy
import numpy as np
from utils.text_manager import generate_key
from utils.cache_manager import save_cache, load_cache, get_cache_keys
from utils.log_manager import get_logger

logger = get_logger(__name__)
def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CacheOpenAI:
    def __init__(self, api_key="empty",
                 base_url="https://api.deepseek.com",
                 model='deepseek-chat',
                 output_root='deepseek-chat',
                 cache_filename='cache',
                 cache_table='task_table'):
        # 初始化大模型接口
        # model名，curl http://localhost:8000/v1/models 查看
        self.model = model
        # ======缓存信息========
        self.cache_root = os.path.join(output_root, self.model)
        os.makedirs(self.cache_root, exist_ok=True)
        self.cache_filename = cache_filename
        self.cache_table_name = cache_table  # 缓存存入的表明
        # 获取表中所有key，临时存储便于筛选非cache数据


        # ==============LLM配置===================
        self.llm_config = {}
        self.llm_config['llm_name'] = self.model
        self.llm_config['llm_base_url'] = base_url
        self.llm_config['generate_params'] = {
            "model": self.model,
            "max_completion_tokens": 2048,
            "n": 1,
            "seed": 0,
            "temperature": 0.0,                 # T>0时,n才能设置>1
            "logprobs": False,                  # 模型会记录生成每个输出 token 时概率最高的 5 个 token 的对数概率。
            "presence_penalty": 1.5,
        }
        self.max_retries = 5                    # API错误最大调用次数
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        client = httpx.Client(limits=limits, timeout=httpx.Timeout(5 * 60, read=5 * 60))
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=client, max_retries=2)
        self.cache_keys = set(get_cache_keys(self.cache_root, self.cache_filename, self.cache_table_name))
    def get_model_max_tokens(self):
        try:
            models = self.client.models.list()
            for model_info in models.data:

                if model_info.id == self.model:
                    return int(model_info.max_model_len)
            logger.info(f"找不到该模型: {self.model}")
            return 8192
        except:
            return 8192

    def get_output_max_tokens(self):
        return self.llm_config['generate_params']['max_completion_tokens']

    def get_input_max_tokens(self):
        total_tokens = self.get_model_max_tokens()
        return total_tokens - self.get_output_max_tokens()

    def get_messages(self, user_message, system_prompt=None):
        if system_prompt is None:
            messages = [
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        return messages

    @dynamic_retry_decorator
    def get_answer(self, messages, params):
        '''
        将信息输入到大模型中，获取答案。该函数是用于调用API接口
        '''
        # params['messages'] = messages
        # cur_params = {}
        # for k in params:
        #     # 额外的参数，去除
        #     if k not in ['desc', 'enable_tqdm']:
        #         cur_params[k] = params[k]
        #
        # cur_params["messages"] = messages


        response = self.client.chat.completions.create(messages=messages, **params)
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        enable_logprobs = params.get('logprobs', False)
        if enable_logprobs:
            # 即开启了logprobs
            all_responses = []
            for choice in response.choices:
                logprobs = [token.logprob for token in choice.logprobs.content]
                perplexity_score = np.exp(-np.mean(logprobs))
                finish_reason = choice.finish_reason

                all_responses.append({
                    'response': choice.message.content,
                    'perplexity_score': perplexity_score,
                    'finish_reason': finish_reason,
                })
            # 根据困惑度排序，
            sorted(all_responses, key=lambda x: x['perplexity_score'], reverse=True)

            return {
                'response': all_responses[0]['response'],
                'finish_reason': all_responses[0]['finish_reason'],
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        else:
            return {
                'response': response.choices[0].message.content,
                'finish_reason': response.choices[0].finish_reason,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }

    def batch_infer(self, batch_messages, **kwargs):
        params = deepcopy(self.llm_config['generate_params'])
        if kwargs:
            params.update(kwargs)

        if params['n'] > 1:
            params['logprobs'] = True
        enable_cache = params.pop('enable_cache', True)
        enable_tqdm = params.pop('enable_tqdm', True)
        desc = params.pop('desc', 'LLM处理')
        # 第一步检查内存情况
        idx2hit_key = self.check_cache(batch_messages, params, enable_cache)
        # 存方所有消息的key
        batch_keys = []
        # 第二步，将待处理的消息进入队列进行处理
        sequence_key = []
        sequence_messages = []

        for idx in range(len(batch_messages)):
            hit, key = idx2hit_key[idx]
            batch_keys.append(key)
            if not hit:
                sequence_messages.append(batch_messages[idx])
                sequence_key.append(key)

        if len(sequence_key):
            with ThreadPoolExecutor(max_workers=80) as executor:
                futures = [executor.submit(self.get_answer, messages, params) for messages in sequence_messages]  # 提交任务
                llm_results = [future.result() for future in tqdm(futures, desc=desc, disable=not enable_tqdm)]  # 获取所有任务的结果
            sequence_responses = []
            for response_list in llm_results:
                sequence_responses.append(response_list)

            # 构建缓存数据
            cache_sequence_data = []
            for key, metadata in zip(sequence_key, sequence_responses):
                metadata['key'] = key
                cache_sequence_data.append(metadata)
                self.cache_keys.add(key)
            save_cache(cache_root=self.cache_root,
                       sequences=cache_sequence_data,
                       filename=self.cache_filename,
                       table_name=self.cache_table_name)

        # 然后从缓存文件中检索信息 [{'key', 'response', 'finish_reason', 'prompt_tokens', 'completion_tokens'}]
        batch_results = load_cache(cache_root=self.cache_root,
                                   sequence_key=batch_keys,
                                   filename=self.cache_filename,
                                   table_name=self.cache_table_name)

        return batch_results

    # 使用简单的messages
    def batch_reply(self, batch_prompts, batch_system_prompts=None, **kwargs):
        batch_messages = []
        if batch_system_prompts is None:
            for prompt in batch_prompts:
                batch_messages.append(self.get_messages(prompt))
        else:
            for prompt, system_prompt in zip(batch_prompts, batch_system_prompts):
                batch_messages.append(self.get_messages(prompt, system_prompt))
        batch_results = self.batch_infer(batch_messages, **kwargs)
        batch_responses = []
        for item in batch_results:
            batch_responses.append(item.get('response', 'API ERROR'))
        return batch_responses
    # 对缓存进行操作=============================

    # 对用户的消息进行编码
    def encode_key(self, messages, local_params):
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "params": local_params,
            "model": self.model,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = generate_key(key_str, prefix='llm-')
        return key_hash

    # idx: message
    # 检测消息列表中，哪些在缓存中
    def check_cache(self, messages_list, params, enable_cache=True):
        key_list = [self.encode_key(messages, params) for messages in messages_list]
        key2idxs = defaultdict(set)  # key转原本序列号
        idx2hit_key = {}  # 序列号对应的cache状态

        for idx, key in enumerate(key_list):
            key2idxs[key].add(idx)
            idx2hit_key[idx] = (False, key)
        if enable_cache:
            key_set = set(key2idxs.keys())
            keys_in_cache = self.cache_keys & key_set
            for key in keys_in_cache:
                for idx in key2idxs[key]:
                    idx2hit_key[idx] = (True, key)

        return idx2hit_key


# 这是一个简陋的无缓存API模型
class APIModel:
    def __init__(self, api_key="sk-e2b55765c7224baab6842f4053b5033e", base_url="https://api.deepseek.com", model='deepseek-chat'):
        # 初始化大模型接口
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def get_answer(self, messages,  max_error=5):
        '''
        将信息输入到大模型中，获取答案。该函数是用于调用API接口，max_error即最大错误次数，一旦一个问题过了这个次数，就判断为错误生成
        :param messages: 用户信息[信息1,...]
        :param tools: 是否启用搜索引擎等工具
        :param max_error: 调用错误次数
        :return: LLM答案
        '''
        for attempt in range(max_error):
            # try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        return None

    def get_messages(self, user_message, system_prompt=None):
        # 将用户输入处理成适合大模型的格式
        messages = [ {
            "role": "user",
            "content": user_message,
        }]
        return messages

    def reply(self, user_message, system_prompt=None):
        # 流式处理，一次处理一个问题
        messages = self.get_messages(user_message, system_prompt)
        answer = self.get_answer(messages)
        return answer

    def batch_reply(self, user_message_list, system_prompt=None, desc='LLM处理'):
        # 并行处理，一次处理多个问题。但由于API问题，只能用流式处理伪造并行处理
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(self.reply, user_message, system_prompt) for user_message in user_message_list]  # 提交任务
            results = [future.result() for future in tqdm(futures, desc=desc)]  # 获取所有任务的结果
        return results