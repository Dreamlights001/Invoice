# 发票信息抽取项目

本项目用于从PDF或图片格式发票中抽取结构化字段信息，支持OCR、文本分块、字段解析、结果合并、视觉反思和字段级评估。项目提供基础方法、RAG增强方法、视觉反思方法以及联合方法，可用于方法验证、对照实验和批量处理。

## 功能特性

- 支持单文件处理和目录批处理
- 支持 `base`、`rag_only`、`reflect_only`、`rag_reflect` 四种方法
- 支持 OCR 缓存复用
- 支持基于 `truth` 的字段级评估
- 支持 Excel 和 CSV 对比报表导出
- 支持 RAG 知识库和 few-shot 预计算结果生成

## 项目结构

```
invoice-hjc/
├── config.json                 # 配置文件
├── main.py                     # 单文件处理入口
├── batch_process.py            # 批处理入口
├── generate_comparison.py      # 生成单方法对比报表
├── compare_methods.py          # 汇总多方法实验结果
├── prepare_rag_assets.py       # 构建RAG知识库
├── precompute_fewshot.py       # 生成few-shot预计算结果
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明文档
├── node/
│   ├── ocr_node.py             # OCR节点
│   ├── chunk_node.py           # 文本分块节点
│   ├── parse_block_node.py     # 分块解析节点
│   ├── merge_node.py           # 分块结果合并
│   ├── reflect_node.py         # 视觉反思节点
│   ├── evaluate_node.py        # 评估节点
│   └── rag_store.py            # RAG检索封装
├── input/
│   ├── invoices/               # 原始发票样本
│   └── truth/                  # truth JSON标注
├── interVar/                   # OCR缓存目录
├── knowledge/                  # RAG知识库目录
├── fewshot_precomputed/        # few-shot预计算结果目录
└── output/                     # 输出目录
```

## 环境配置

### Python环境

建议使用 Python 3.9 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

### api配置

项目通过本地 `config.json` 读取接口与模型配置。

当前使用的主要配置项包括：

```json
{
  "LLM_API_URL": "",
  "LLM_API_KEY": "",
  "LLM_MODEL": "",
  "reflect_LLM_API_URL": "",
  "reflect_LLM_API_KEY": "",
  "reflect_LLM_MODEL": "",
  "EMBEDDING_URL": "",
  "EMBEDDING_API_KEY": "",
  "EMBEDDING_MODEL": "",
  "rerank_url": "",
  "rerank_api": "",
  "rerank_model": "",
  "ocr_base_url": "",
  "ocr_model": "",
  "ocr_api_key": ""
}
```

说明：

- `LLM_*` 用于分块、块解析和合并阶段
- `reflect_LLM_*` 用于视觉反思节点


## 使用示例

### RAG

RAG 相关文件包括：

- `knowledge/knowledge.sqlite`
- `fewshot_precomputed/`

建立RAG知识库：

```bash
python .\prepare_rag_assets.py
```

生成预计算结果：

```bash
python .\precompute_fewshot.py --input .\interVar --output .\fewshot_precomputed --enable_rerank
```

### 单文件处理

```bash
python .\main.py --input .\input\invoices\invoice.PDF --truth .\input\truth --output .\output\pending --method base
```

```bash
python .\main.py --input .\input\invoices\invoice.PDF --truth .\input\truth --intervar .\interVar --output .\output\pending --method rag_reflect
```

### 批量处理

```bash
python .\batch_process.py --input .\input\invoices --truth .\input\truth --intervar .\interVar --output .\output\pending --method rag_reflect --workers 4 --skip_existing
```

### 生成单方法对比表

```bash
python .\generate_comparison.py --truth .\input\truth --output .\output\pending\rag_reflect --method rag_reflect --out_file .\output\pending\rag_reflect\comparison.rag_reflect.xlsx
```

### 汇总多方法实验结果

```bash
python .\compare_methods.py --base_output .\output\ac --experiment_output .\output\pending --out_file .\output\pending\method_comparison.xlsx
```


## 输入与输出

### 输入

- `input/invoices/`：原始发票文件
- `input/truth/`：对应的 truth JSON
- `interVar/`：OCR 缓存目录，可选

### 输出

不同方法默认会写入不同子目录，避免结果互相覆盖。

以 `output/pending` 为例：

- `base`：`output/pending/<文件名>/`
- `rag_only`：`output/pending/rag_only_<文件名>/`
- `reflect_only`：`output/pending/reflect_only_<文件名>/`
- `rag_reflect`：`output/pending/rag_reflect_<文件名>/`

每个样本目录中通常包含：

- `*.blocks.json`
- `*.block_parses.json`
- `*.merged.json`
- `*.reflected.json`
- `*.invoice.json`
- `*.evaluation.json`
- `*.metadata.json`
- `prompts/`

## OCR 缓存机制

不传 `--intervar` 时：

- 不读取缓存
- 不保存缓存
- 直接调用 OCR 接口

传入 `--intervar` 时：

- 优先读取 `interVar/*.txt`
- 命中则直接复用
- 未命中则调用 OCR 并写入缓存目录

## 处理方法

### `base`

使用 OCR 文本完成分块、块解析、整单合并与评估。

### `rag_only`

在分块和块解析阶段引入 RAG few-shot 检索结果。

### `reflect_only`

在合并阶段之后引入视觉反思节点，结合原始图像或 PDF 对整单结果进行核验与纠错。

### `rag_reflect`

联合使用 RAG 与视觉反思。