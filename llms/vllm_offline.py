'''
功能简介：在线LLM API调用

加入了sqlite缓存机制，相同参数和提示词的则会调用缓存
'''
from vllm.lora.request import LoRARequest
from collections import defaultdict
import os
import json
from copy import deepcopy
from utils.text_manager import generate_key
from utils.cache_manager import save_cache, load_cache, get_cache_keys
from utils.log_manager import get_logger
import vllm


logger = get_logger(__name__)
class CacheVllm:
    def __init__(self,
                 model_path='deepseek-chat',
                 output_root='./dataset_name/llm_cache',
                 cache_filename='cache',
                 cache_table='task_table', **kwargs):

        # ======缓存信息========
        self.model_path = model_path
        self.model_name = os.path.basename(self.model_path)
        self.cache_root = os.path.join(output_root, self.model_name)
        self.cache_filename = cache_filename
        self.cache_table_name = cache_table  # 缓存存入的表明

        self.lora_path_dict = kwargs.get('lora_path_dict', {})
        self.lora_name2path = {}
        lora_modules = []
        # ==============LLM配置===================
        if len(self.lora_path_dict) == 0:
            self.enable_lora = False
        else:
            self.enable_lora = True
            for idx, (lora_name, lora_path) in enumerate(self.lora_path_dict.items()):
                self.lora_name2path[lora_name] = (idx, lora_path)

        self.client = vllm.LLM(self.model_path,
                               tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),      # 控制使用多少块 GPU 进行张量并行推理
                               pipeline_parallel_size=kwargs.get('pipeline_parallel_size', 1),  # 设置流水线并行的阶段数（stage）。适用于非常大的模型，如 GPT-3 或 Llama-65B+。
                               seed=kwargs.get('seed', 123),
                               dtype='auto',
                               # enable_prefix_caching=True,
                               gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.5),
                               quantization=kwargs.get('quantization', None),
                               load_format=kwargs.get('load_format', 'auto'),
                               trust_remote_code=kwargs.get('trust_remote_code', True),
                               enforce_eager=False,
                               enable_lora=self.enable_lora,
                               )

        self.tokenizer = self.client.get_tokenizer()

        # vllm配置
        self.global_params = kwargs
        self.global_params['model_name'] = self.model_name
        # 生成配置
        self.sampling_params = {'n':1,
                                'max_tokens': 2048,
                                'temperature': 0,
                                'skip_special_tokens': True}

        self.cache_keys = set(get_cache_keys(self.cache_root, self.cache_filename, self.cache_table_name))
    def get_model_max_tokens(self):
        return self.tokenizer.model_max_length

    def get_output_max_tokens(self):
        return self.sampling_params['max_tokens']

    def get_input_max_tokens(self):
        total_tokens = self.get_model_max_tokens()
        return total_tokens - self.get_output_max_tokens()

    def convert_message_to_prompt(self, user_message):
        prompt = self.tokenizer.apply_chat_template(conversation=user_message,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        return prompt

    def batch_get_answer(self, batch_prompts, params, enable_tqdm=False):
        sampling_params = [vllm.SamplingParams(**params)]*len(batch_prompts)
        if self.enable_lora:
            lora_name = params.get('lora_name', None)
            if lora_name is None:
                batch_responses = self.client.generate(
                    batch_prompts,
                    sampling_params,
                    use_tqdm=enable_tqdm,
                )
            else:
                lora_id, lora_path = self.lora_name2path[lora_name]
                batch_responses = self.client.generate(batch_prompts, sampling_params,
                                                 use_tqdm=enable_tqdm,
                                                 lora_request=LoRARequest(lora_name, lora_id, lora_path)
                                                 )
        else:
            batch_responses = self.client.generate(
                batch_prompts,
                sampling_params,
                use_tqdm=enable_tqdm,
            )
        batch_results = []
        for vllm_output in batch_responses:
            batch_results.append({
                'response': vllm_output.outputs[0].text,
                'finish_reason': vllm_output.outputs[0].finish_reason,
                'prompt_tokens': len(vllm_output.prompt_token_ids),
                'completion_tokens': len(vllm_output.outputs[0].token_ids)
            })
        return batch_results


    def batch_infer(self, batch_messages, **kwargs):
        params = deepcopy(self.sampling_params)
        if kwargs:
            params.update(kwargs)

        enable_cache = params.pop('enable_cache', True)
        max_completion_tokens = params.pop('max_completion_tokens', -1)
        if max_completion_tokens > 0:
            params['max_tokens'] = max_completion_tokens
        batch_prompts = []
        for message in batch_messages:
            batch_prompts.append(self.convert_message_to_prompt(message))

        enable_tqdm = params.pop('enable_tqdm', False)
        desc = params.pop('desc', None)
        # 第一步检查内存情况
        idx2hit_key = self.check_cache(batch_prompts, params, enable_cache)
        # 存方所有消息的key
        batch_keys = []
        # 第二步，将待处理的消息进入队列进行处理
        sequence_key = []
        sequence_prompt = []

        for idx in range(len(batch_prompts)):
            hit, key = idx2hit_key[idx]
            batch_keys.append(key)
            if not hit:
                sequence_prompt.append(batch_prompts[idx])
                sequence_key.append(key)
        if len(sequence_key):
            sequence_response = self.batch_get_answer(sequence_prompt, params, enable_tqdm)
            # 缓存文件
            cache_sequence_data = []
            for key, metadata in zip(sequence_key, sequence_response):
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
        # print(batch_results)
        batch_responses = []
        for item in batch_results:
            batch_responses.append(item.get('response', 'API ERROR'))
        return batch_responses
    # 对缓存进行操作=============================

    # 对用户的消息进行编码
    def encode_key(self, prompt, local_params):
        key_data = {
            "prompt": prompt,  # messages requires JSON serializable
            'local_params': local_params,
            'global_params': self.global_params,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = generate_key(key_str, prefix='llm-')
        return key_hash
    # idx: message
    # 检测消息列表中，哪些在缓存中
    def check_cache(self, batch_prompts, params, enable_cache=True):
        key_list = [self.encode_key(prompt, params) for prompt in batch_prompts]
        key2idxs = defaultdict(set)  # key转原本序列号
        idx2hit_key = {} # 序列号对应的cache状态

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