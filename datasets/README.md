在此致谢[HippoRAG 2](https://github.com/ianliuwd/HippoRAG2)作者提供的数据；可以点击[链接]([osunlp/HippoRAG_2 · Datasets at Hugging Face](https://huggingface.co/datasets/osunlp/HippoRAG_2))获取数据。输入下载后，如下存放：

```
- datasets/
	- hotpotqa/
		- hotpotqa.json
		- hotpotqa_corpus.json
```



如果需要添加数据，可以在utils/dataset_manager.py中添加处理函数，在main.py中处理数据集。



---

We thank the authors of [HippoRAG 2](https://github.com/ianliuwd/HippoRAG2) for sharing the data.  
You can download it from this [link](https://huggingface.co/datasets/osunlp/HippoRAG_2). After downloading, place the files as follows:

```
- datasets/
  - hotpotqa/
    - hotpotqa.json
    - hotpotqa_corpus.json
```

To add new datasets, implement the corresponding processing function in `utils/dataset_manager.py` and register it in `main.py`.