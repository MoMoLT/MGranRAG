import sys

from langchain.embeddings.base import Embeddings
# 自定义 Embedding 类，适配 LangChain 接口
from transformers import AutoModel
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List
import torch
import numpy as np
# {'Recall@1': 0.4705, 'Recall@2': 0.8575, 'Recall@5': 0.9535, 'Recall@10': 0.9785, 'Recall@20': 0.987, 'Recall@30': 0.9895, 'Recall@50': 0.9935, 'Recall@100': 0.9955, 'Recall@150': 0.9975, 'Recall@200': 0.998}
def get_query_instruction(linking_method):
    instructions = {
        'phrase_to_phrase': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_doc': 'Given a multi-hop question, retrieve relevant documents that can help answer the complex question.',
        'agument_query_to_doc': 'Given a multi-hop question and some relevant knowledges, retrieve relevant documents that can help answer the complex question.',
    }#0.9535
    instructionsv2 = {
        'phrase_to_phrase': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_doc': 'Given a multi-hop question, retrieve documents that can help answer the question',
    }# 0.9515
    instructionsv1 = {
        'phrase_to_phrase': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_doc': 'Given a question, retrieve relevant documents that best answer the question.',
    }# 0.91
    default_instruction = instructions['query_to_doc']
    return instructions.get(linking_method, default_instruction)

# class NVEmbedding(Embeddings):
#     def __init__(self,
#                  model_name: str = "NeuralVenture/NV-Embed-v2",
#                  encode_kwargs: dict = None,
#                  batch_size: int = 16):
#         super().__init__()
#         self.batch_size = batch_size
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # 半精度、自动分配到 GPU/CPU
#         hf_kwargs = {
#             "trust_remote_code": True,
#             "device_map": "auto",
#             "torch_dtype": torch.float16,
#             "low_cpu_mem_usage": True,
#         }
#         self.model = AutoModel.from_pretrained(model_name, **hf_kwargs).eval()
#         # 你自己的 encode 参数，比如 max_length、do_tokenize 之类
#         self.encode_params = encode_kwargs or {"max_length": 512}
#         print("NVEmbedding loaded on", self.device, "with params", self.encode_params

class NVEmbedding(Embeddings):
    """LangChain 风格的 Embeddings 接口，内部加载 NV-Embed-v2，支持半精度、自动 offload"""

    def __init__(self,
                 model_name: str = "NeuralVenture/NV-Embed-v2",
                 encode_kwargs: dict = None,
                 batch_size: int = 16):
        super().__init__()
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 半精度、自动分配到 GPU/CPU
        hf_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModel.from_pretrained(model_name, **hf_kwargs).eval()
        # 你自己的 encode 参数，比如 max_length、do_tokenize 之类
        self.encode_params = encode_kwargs or {"max_length": 512}
        print("NVEmbedding loaded on", self.device, "with params", self.encode_params)

    def embed_documents(self, texts):
        """批量对文档做向量化，返回 List[List[float]]"""
        emb_tensor = self._batch_encode(texts, mode="doc")
        return emb_tensor.cpu().numpy().tolist()

    def embed_query(self, text, mode: str='doc'):
        """对单条 query 做向量化，返回 List[float]"""
        emb_tensor = self._batch_encode([text], mode=mode)
        return emb_tensor[0].cpu().numpy().tolist()

    def _batch_encode(self, texts, mode: str):
        """
        内部 batch encode，返回归一化后的 torch.Tensor，shape = (len(texts), dim)
        mode: 'doc' | 'query_to_doc' | 'entity_to_entity'
        """
        # assert mode in ("doc", "query_to_doc", "phrase_to_phrase")
        # instr = ""
        # if mode != "doc":
        #     # doc 模式下不加 instruction
        #     instr = get_query_instruction(mode)

        results = []
        with torch.no_grad():
            base_params = deepcopy(self.encode_params)
            if mode != "doc":
                instr = get_query_instruction(mode)
                if mode == 'agument_query_to_doc':
                    base_params["instruction"] = f"Instruct: {instr}\nKnowledges: "
                else:
                    base_params["instruction"] = f"Instruct: {instr}\nQuery: "
            else:
                base_params["instruction"] = ""
            # if len(instr):
            #     base_params["instruction"] = f"Instruct: {instr}\nQuery: "
            # else:
            #     base_params["instruction"] = ''

            # 根据 batch_size 分批
            for i in tqdm(range(0, len(texts), self.batch_size), desc="NVEmbedding batching", disable=True):
                slice_ = texts[i : i + self.batch_size]
                base_params["prompts"] = slice_
                # 假设 huggingface 实现里有个 encode(**params) 方法
                emb = self.model.encode(**base_params)    # [bsz, dim]
                # half precision 时要先搬到 GPU，再归一化
                emb = emb.to(self.device)
                emb = F.normalize(emb, p=2, dim=1)
                results.append(emb.cpu())

        return torch.cat(results, dim=0)