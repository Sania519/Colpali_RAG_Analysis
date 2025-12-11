# ColPali RAG System

A comprehensive implementation and evaluation of two-stage retrieval systems for document Q&A, comparing **ColPali** (vision-based) and **Text-based** (OCR + embeddings) approaches on the Open RAGBench dataset.

## Overview

This repository implements:
- **Two-Stage Retrieval Architecture**: Stage 1 retrieves relevant documents, Stage 2 retrieves specific pages
- **ColPali System**: Vision-language model with late interaction mechanism
- **Text-Based Baseline**: Traditional OCR + text embeddings approach
- **Comprehensive Evaluation**: Accuracy metrics (NDCG@5, Recall@1/5, MRR) and latency benchmarks (P95/P99)

## Features

### Core Systems
- **ColPali RAG**: Multi-vector representations with late interaction scoring
- **Text-Based RAG**: Sentence transformer embeddings with cosine similarity
- **Configurable Two-Stage Pipeline**: Adjustable top-k for both stages
- **Multiple Aggregation Strategies**: Mean, max, and weighted pooling for document embeddings

### Evaluation Tools
- **Accuracy Metrics**: NDCG@5, Recall@1/5, MRR, stage-specific metrics
- **Latency Profiling**: P50/P95/P99 measurements for all operations
- **System Comparison**: Direct performance comparisons between approaches

## Installation

```bash
pip install torch colpali-engine datasets transformers pillow numpy tqdm pymupdf requests
```

For CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Download Dataset

```python
from open_ragbench_loader import OpenRAGBenchLoader

loader = OpenRAGBenchLoader()
loader.download_dataset(max_queries=50)
loader.load_dataset()
```

### 2. Run ColPali System

```python
from two_stage_colpali import TwoStageColPaliRAG, PDFProcessor, OpenRAGBenchDataset

pdf_processor = PDFProcessor(pdf_dir="./arxiv_pdfs")
dataset = OpenRAGBenchDataset(loader, pdf_processor, max_queries=50)

colpali_rag = TwoStageColPaliRAG("vidore/colpali-v1.2")
colpali_rag.index_documents(dataset.arxiv_ids, dataset.doc_to_pages, aggregation="max")

query_embeddings = colpali_rag.encode_queries(dataset.queries)
results = colpali_rag.retrieve_two_stage(query_embeddings, top_k_docs=20, top_k_pages=20, rerank=True)
```

### 3. Run Text-Based Baseline

```python
from text_based_rag import TextBasedRAG, PDFTextExtractor, TextBasedDataset, TextEmbeddingModel

text_extractor = PDFTextExtractor(pdf_dir="./arxiv_pdfs")
text_dataset = TextBasedDataset(loader, text_extractor, max_queries=50)

embedding_model = TextEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
text_rag = TextBasedRAG(embedding_model)
text_rag.index_documents(text_dataset.arxiv_ids, text_dataset.doc_to_page_texts)

results = text_rag.retrieve_two_stage(text_dataset.queries, top_k_docs=20, top_k_pages=20)
```

### 4. Run Latency Benchmark

```python
from latency_benchmark import run_full_benchmark

colpali_metrics, text_metrics = run_full_benchmark(
    colpali_rag=colpali_rag,
    colpali_dataset=colpali_dataset,
    text_rag=text_rag,
    text_dataset=text_dataset,
    num_queries=100,
    num_docs=20
)
```

## File Structure

```
├── colpali_rag_system.ipynb      # Complete Jupyter notebook with all implementations
└── README.md                     # This file
```

## Notebook Structure

The notebook is organized into the following sections:

1. **Setup & Dependencies** - Installation and imports
2. **Basic ColPali Demo** - Simple evaluation on ArxivQA dataset
3. **Dataset Loader** - OpenRAGBenchLoader class implementation
4. **Two-Stage ColPali RAG** - Full implementation with document/page retrieval
5. **Text-Based RAG Baseline** - OCR + text embeddings approach
6. **Latency Benchmark** - Performance profiling tools
7. **Full Benchmark Execution** - Running complete comparison

## Configuration Parameters

### ColPali System
```python
TOP_K_DOCS = 20              # Documents to retrieve in Stage 1
TOP_K_PAGES = 20             # Pages to retrieve in Stage 2
AGGREGATION = "max"          # Document embedding strategy: "mean", "max", or "weighted"
RERANK = True                # Global re-ranking of candidate pages
BATCH_SIZE = 4               # Encoding batch size
```

### Text-Based System
```python
TOP_K_DOCS = 20              # Documents to retrieve in Stage 1
TOP_K_PAGES = 20             # Pages to retrieve in Stage 2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 512             # Token truncation limit
BATCH_SIZE = 32              # Encoding batch size
```

## Evaluation Metrics

### Accuracy Metrics
- **NDCG@5**: Normalized Discounted Cumulative Gain at rank 5
- **Recall@1**: Percentage of queries with correct page in top-1
- **Recall@5**: Percentage of queries with correct page in top-5
- **MRR**: Mean Reciprocal Rank
- **Doc_Recall@5**: Stage 1 document retrieval accuracy
- **Page_Given_Doc_Recall**: Stage 2 page retrieval accuracy (given correct document)
- **End_to_End_Success**: Both document and page must be correct

### Latency Metrics
- **P50, P95, P99 Latencies**: Percentile measurements
- **Mean/Median/Std**: Statistical distributions
- Query encoding, document encoding, retrieval operations
- End-to-end pipeline measurements

## Performance Tips

### Improving Accuracy
1. Increase `TOP_K_DOCS` (10→20) for better Stage 1 recall
2. Use `aggregation="max"` to capture salient content
3. Enable `rerank=True` for global re-ranking
4. Increase `TOP_K_PAGES` for evaluation

### Optimizing Latency
1. Batch query/document encoding
2. Cache document embeddings
3. Use smaller models for faster inference
4. Adjust batch sizes based on GPU memory

## Dataset

**Open RAGBench** (Vectara)
- Scientific papers from arXiv
- Multi-page document retrieval
- Diverse query types
- Page-level ground truth annotations

## Model Architectures

### ColPali
- Vision-language model based on PaliGemma
- Late interaction mechanism for fine-grained matching
- Multi-vector representations
- Direct image understanding without OCR

### Text-Based
- Sentence transformers for text encoding
- PyMuPDF for PDF text extraction
- Cosine similarity for matching
- Single-vector per page

## System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- Storage for PDFs and model weights

## Citation

If you use ColPali in your research:
```bibtex
@article{colpali2024,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and others},
  journal={arXiv preprint arXiv:2407.01449},
  year={2024}
}
```

## License

This project is provided for research and educational purposes. Please check the licenses of individual models and datasets used.

## Acknowledgments

- ColPali model from ViDoRe benchmark
- Open RAGBench dataset from Vectara
- Sentence Transformers library
- PyMuPDF for PDF processing
