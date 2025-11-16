# MGranRAG 系统部署与使用手册
如果你无法连接huggingface,请在Linux输入：
```angular2html
export HF_ENDPOINT=https://hf-mirror.com
```
本手册详细说明了 MGranRAG 系统的完整环境搭建、数据集配置、模型部署及运行流程。由于 NV-Embed-v2 与 Qwen3 所需运行环境存在依赖冲突，同时为支持大模型（如 Llama3.3-70B）在 80GB 内存下的运行，系统采用双环境隔离架构：一个用于 LLM 服务，另一个用于 RAG 任务执行。

## LLM 服务环境搭建

创建独立的 Conda 虚拟环境用于部署大型语言模型（LLM），推荐使用 Python 3.10。

```bash
conda create -n llm_serve python=3.10
conda activate llm_serve
pip install vllm
```

### 部署 Qwen3-8B 模型服务

在 `llm_serve` 环境中启动基于 vLLM 的推理服务：

```bash
nohup vllm serve Qwen3-8B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --port 8000 > serve.log 2>&1 &
```

查看服务日志以确认模型加载状态：

```bash
tail -f serve.log
```

### 安装并配置 Ollama 环境

Ollama 用于运行 Llama3.3-70B 等超大规模模型，需单独安装：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

启动 Ollama 服务并加载 Llama3.3-70B 模型：

```bash
ollama serve &
ollama run llama3.3:70b
```

> **注意**：请确保系统具备足够的 GPU 显存以支持 70B 模型运行。

---

## MGranRAG 运行环境搭建

创建独立环境用于运行MGranRAG。

```bash
conda create -n mgranrag python=3.10
conda activate mgranrag
pip install -r requirements.txt
```

安装必要的 spaCy 英文语言模型：

```bash
python -m spacy download en_core_web_sm
```

---

## 数据集配置

本系统基于HippoRAG 2的公开数据集进行构建。建议将数据集按如下目录结构组织：

```
MGranRAG/
├── datasets/
│   ├── hotpotqa/
│   │   ├── hotpotqa.json
│   │   └── hotpotqa_corpus.json
│   ├── musique/
│   │   ├── musique.json
│   │   └── musique_corpus.json
│   └── ...
├── llms/
├── models/
└── ...
```

请确保 `datasets` 目录下包含对应任务的数据文件，并保持命名一致。

---

## 核心参数配置

主要配置项位于 `config.py` 文件中，关键参数如下：

### 1. RAG 模型配置

```python
class rag:
    # embedding_path = '/root/autodl-tmp/NV-Embed-v2'  # NV-Embed-v2 路径
    embedding_path = '/root/autodl-fs/contriever-msmarco'  # 或 Contriever 等通用 embedding 模型路径
    model_mode = 'normal'  # 'instruction' if using NV-Embed-v2; 'normal' otherwise
    max_embedding_length = 2048
```

- 若使用 **NV-Embed-v2**，请设置 `model_mode = 'instruction'` 并提供完整的 prompt 指令。
- 对于常规 embedding 模型（如 Contriever、all-MiniLM-L6-v2），使用 `model_mode = 'normal'`。

### 2. LLM 接口配置

```python
class llm:
    api_key = 'empty'
    base_url = 'http://127.0.0.1:8000/v1'  # vLLM 服务地址
    model = 'Qwen3-8B'
    model_path = '/root/autodl-tmp/Qwen3-8B'
    enable_online = True  # 启用在线检索功能
```

---

## 功能运行指南

### 启动 LLM 服务
#### 使用 vLLM 部署 Qwen3-8B（用于检索阶段）

```bash
conda activate llm_serve
nohup vllm serve Qwen3-8B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --port 8000 > serve.log 2>&1 &
tail -f serve.log
```

#### 切换至 Ollama 运行 Llama3.3-70B（用于问答阶段）
关闭当前 vLLM 服务后，启动 Ollama：

```bash
pkill -f "vllm"
nohup ollama serve > serve.log 2>&1 &
tail -f serve.log
```

---

### 执行检索任务

激活 MGranRAG 环境并运行检索流程：

```bash
conda activate mgranrag
python main.py --data hotpotqa --mode retrieval
```

此步骤将使用配置的 embedding 模型对文档进行编码，并构建向量索引。

---

### 执行问答任务

在执行前，若GPU内存资源不足，可以将Qwen3-8B服务关闭，再使用 Ollama 提供的 Llama3.3-70B 模型进行最终答案生成：

```bash
python main.py \
  --data hotpotqa \
  --mode qa \
  --url http://localhost:11434/v1 \
  --model_name llama3.3:70b \
  --model_path [Your_Llama3_3_70B_Tokenizer_Path]
```

> 请替换 `[Your_Llama3_3_70B_Tokenizer_Path]` 为实际 tokenizer 文件路径。

---

### 端到端问答流程

若使用单一 LLM 完成检索与问答全过程，可直接运行：

```bash
python main.py --data hotpotqa --mode end2end
```

该模式下系统将自动调度检索与生成流程，适用于测试和快速验证。