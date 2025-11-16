# MGranRAG System Deployment and User Manual

This manual provides a comprehensive guide to the setup, dataset configuration, model deployment, and execution workflow for the MGranRAG system. Due to dependency conflicts between NV-Embed-v2 and Qwen3, and to support large models (e.g., Llama3.3-70B) under 80GB of system memory, the system adopts a dual-environment architecture: one dedicated to LLM serving and another for RAG task execution.

---

## LLM Serving Environment Setup

Create an isolated Conda virtual environment for deploying large language models (LLMs), using Python 3.10.

```bash
conda create -n llm_serve python=3.10
conda activate llm_serve
pip install vllm
```

### Deploy Qwen3-8B Model Service

Start the vLLM-based inference server within the `llm_serve` environment:

```bash
nohup vllm serve Qwen3-8B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --port 8000 > serve.log 2>&1 &
```

Monitor the service logs to verify model loading status:

```bash
tail -f serve.log
```

### Install and Configure Ollama

Ollama is used to run ultra-large models such as Llama3.3-70B and must be installed separately:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama service and load the Llama3.3-70B model:

```bash
ollama serve &
ollama run llama3.3:70b
```

> **Note**: Ensure sufficient GPU memory is available to support the 70B model.

---

## MGranRAG Runtime Environment Setup

Create a separate environment for running MGranRAG components:

```bash
conda create -n mgranrag python=3.10
conda activate mgranrag
pip install -r requirements.txt
```

Install the required spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

---

## Dataset Configuration

This system is built upon the public dataset from [HippoRAG 2](https://github.com/hipporag/hipporag). It is recommended to organize the datasets in the following directory structure:

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

Ensure that the `datasets` directory contains the corresponding data files with consistent naming.

---

## Core Configuration Parameters

Key configuration parameters are defined in the `config.py` file.

### 1. RAG Model Configuration

```python
class rag:
    embedding_path = '/root/autodl-tmp/NV-Embed-v2'  # Path to NV-Embed-v2
    # embedding_path = '/root/autodl-fs/contriever-msmarco'  # Path to general embedding models (e.g., Contriever)
    model_mode = 'instruction'  # Use 'instruction' for NV-Embed-v2; 'normal' otherwise
    max_embedding_length = 2048
```

- If using **NV-Embed-v2**, set `model_mode = 'instruction'` to enable prompt-augmented text encoding.
- For standard embedding models (e.g., Contriever, all-MiniLM-L6-v2), use `model_mode = 'normal'`.

### 2. LLM Interface Configuration

```python
class llm:
    api_key = 'empty'
    base_url = 'http://127.0.0.1:8000/v1'  # vLLM service endpoint
    model = 'Qwen3-8B'
    model_path = '/root/autodl-tmp/Qwen3-8B'
    enable_online = True  # Enable online retrieval
```

---

## Functional Execution Guide

### Start LLM Services

#### Deploy Qwen3-8B with vLLM (for retrieval phase)

```bash
conda activate llm_serve
nohup vllm serve Qwen3-8B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --port 8000 > serve.log 2>&1 &
tail -f serve.log
```

#### Switch to Ollama for Llama3.3-70B (for QA phase)

Use `kill` to terminate the current vLLM service and start Ollama:
```bash
ps ux
kill [Your PID]
```

```bash
nohup ollama serve > serve.log 2>&1 &
tail -f serve.log
```

---

### Execute Retrieval Task

Activate the MGranRAG environment and run the retrieval pipeline:

```bash
conda activate mgranrag
python main.py --data hotpotqa --mode retrieval --epoch 1
```

This step encodes documents using the configured embedding model and builds a vector index.

---

### Execute Question Answering Task

Before execution, if GPU memory is limited, stop the Qwen3-8B service and use Ollama’s Llama3.3-70B model to generate final answers:

```bash
python main.py \
  --data hotpotqa \
  --mode qa \
  --url http://localhost:11434/v1 \
  --model_name llama3.3:70b \
  --model_path [Your_Llama3_3_70B_Tokenizer_Path]
```

> Replace `[Your_Llama3_3_70B_Tokenizer_Path]` with the actual path to the tokenizer.

---

### End-to-End QA Pipeline

If using a single default LLM for both retrieval and QA, execute the full pipeline directly:

```bash
python main.py --data hotpotqa --mode end2end  --epoch 1
```

This mode automatically orchestrates retrieval and generation, suitable for testing and rapid validation.