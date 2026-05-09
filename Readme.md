# RAG (Retrieval-Augmented Generation) System Development in Practice

This project is the code repository for a DeepSeek-based RAG system [practical course](https://u.geekbang.org/subject/airag/1009927), implementing a complete Retrieval-Augmented Generation system.

Course link: [RAG System Development in Practice](https://u.geekbang.org/subject/airag/1009927)

![DeepSeek-based RAG System Development Course Architecture Diagram](92-Pic/RAG.PNG)


## Technology Framework

![RAG Technology Framework](92-Pic/RAG-tech-framework.png)

![RAG in Action Book](92-Pic/RAG-in-action.jpg)

> The companion book [*RAG in Action*](https://item.jd.com/14447967.html) has been published! (Posts & Telecom Press)

[Click here to purchase at a discount](https://item.jd.com/14447967.html)

## Project Architecture

The project uses a modular design, with each module responsible for a different aspect of the RAG system:

| Module | Function | Tech Stack | Dependencies |
|--------|----------|------------|--------------|
| `00-SimpleRAG` | Basic RAG system implementation | LangChain/LlamaIndex | Base environment |
| `01-DataLoading` | Data loading and preprocessing | pandas, PyPDF2 | Document parsing libraries |
| `02-DocChunking` | Document chunking strategies | LangChain Splitters | NLP tools |
| `03-Embedding` | Text vectorization | HuggingFace, BGE | GPU support (optional) |
| `04-VectorDB` | Vector database operations | Milvus, Chroma | Vector database |
| `05-PreRetrieval` | Retrieval optimization | Query Expansion | NLP tools |
| `06-Indexing` | Index construction and optimization | Hierarchical index, Keyword index | Search engine |
| `07-PostRetrieval` | Retrieval result optimization | Reranking, Filtering | ML models |
| `08-Generation` | Answer generation | LLM integration | GPU recommended |
| `09-Evaluation` | System performance evaluation | RAGAS, TruLens | Evaluation frameworks |
| `10-AdvanceRAG` | Advanced RAG technique implementation | Graph RAG, Multi-Agent | Advanced frameworks |

## Environment Requirements

### Hardware Requirements

#### GPU Version
- NVIDIA GPU (recommended >= 8GB VRAM)
- CUDA 11.8 or higher
- cuDNN 8.0 or higher

#### CPU Version
- Recommended >= 16GB RAM
- Multi-core processor (recommended >= 4 cores)

### Software Requirements

#### Operating System Support
1. **Ubuntu**
- Recommended: 22.04 LTS

1. **MacOS (Intel/Apple Silicon)**
   - Apple Silicon can use MPS acceleration

2. **Windows 10/11**
   - Recommended: WSL2 + Ubuntu

### Framework Selection

1. Python: 3.10+
2. LangChain Framework
   - Basic version: `requirements_langchain_SimpleRAG(additional-packages-needed-for-later-modules).txt`
   - Full version (GPU): `requirements_langchain_20250413(Ubuntu-with-GPU).txt`
   - Full version (CPU): `requirements_langchain_no-GPU(Mac,Win).txt`

3. LlamaIndex Framework
   - Basic version: `requirements_llamaindex_SimpleRAG(additional-packages-needed-for-later-modules).txt`
   - Full version (GPU): `requirements_llamaindex_20250413(Ubuntu-with-GPU).txt`
   - Full version (CPU): `requirements_llamaindex_no-GPU(Mac,Win).txt`

## Environment Setup

### Ubuntu (GPU Version + LangChain Framework)

```bash
# Create virtual environment
python -m venv venv-rag-langchain
source venv-rag-langchain/bin/activate
## Or use conda
conda create -n venv-rag-langchain python=3.10.12
conda activate venv-rag-langchain

# Install dependencies
pip install -r 91-Environment/requirements_langchain_20250413_Ubuntu-with-GPU.txt
```

### Ubuntu (CPU Version + LangChain Framework)

```bash
# Create virtual environment
python -m venv venv-rag-langchain
source venv-rag-langchain/bin/activate
## Or use conda
conda create -n venv-rag-langchain python=3.10.12
conda activate venv-rag-langchain

# Install dependencies
pip install -r 91-Environment/requirements_langchain_Ubuntu-with-CPU.txt
```

### MacOS/Windows (CPU Version + LangChain Framework)
```bash
# Create virtual environment
python -m venv venv-rag-langchain
# Windows
.\venv-rag-langchain\Scripts\activate
# MacOS
source venv-rag-langchain/bin/activate
## Or use conda
conda create -n venv-rag-langchain python=3.10.12
conda activate venv-rag-langchain

# Install dependencies
pip install -r 91-Environment/requirements_langchain_no-gpu_Mac-Win.txt
```

### Ubuntu (GPU Version + LlamaIndex Framework)

```bash
# Create virtual environment
python -m venv venv-rag-llamaindex
source venv-rag-llamaindex/bin/activate
## Or use conda
conda create -n venv-rag-llamaindex python=3.10.12
conda activate venv-rag-lanllamaindexgchain

# Install dependencies
pip install -r 91-Environment/requirements_llamaindex_20250413_Ubuntu-with-GPU.txt
```

### Ubuntu (CPU Version + LlamaIndex Framework)

```bash
# Create virtual environment
python -m venv venv-rag-llamaindex
source venv-rag-llamaindex/bin/activate
## Or use conda
conda create -n venv-rag-llamaindex python=3.10.12
conda activate venv-rag-llamaindex

# Install dependencies
pip install -r 91-Environment/requirements_llamaindex_Ubuntu-with-CPU.txt
```

### MacOS/Windows (CPU Version + LlamaIndex Framework)
```bash
# Create virtual environment
python -m venv venv-rag-llamaindex
# Windows
.\venv-rag-llamaindex\Scripts\activate
# MacOS
source venv-rag-llamaindex/bin/activate
## Or use conda
conda create -n venv-rag-llamaindex python=3.10.12
conda activate venv-rag-llamaindex

# Install dependencies
pip install -r 91-Environment/requirements_llamaindex_no-gpu_Mac-Win.txt
```

## Special Dependency Notes

1. PDF processing related:
   - Use `requirements_camelot_20250413.txt` to install PDF processing dependencies
   - May require additional system-level dependencies:
     - Ubuntu: `sudo apt-get install ghostscript python3-tk`
     - MacOS: `brew install ghostscript tcl-tk`
     - Windows: Ghostscript must be installed manually

2. Annotation tool related:
   - Use `requirements_marker_20250413.txt` to install annotation tool dependencies

## Usage Instructions

1. Select the appropriate environment configuration file and install dependencies
2. Learn and practice the modules in order
3. Each module contains independent examples and documentation
4. It is recommended to start with `00-SimpleRAG` and progress gradually

## Notes

1. The GPU version requires that the CUDA environment is configured correctly
2. Different operating systems may require additional system-level dependencies
3. It is recommended to use a virtual environment to manage dependencies
4. Some modules may require additional model downloads or API key configuration

## Frequently Asked Questions

1. CUDA-related errors: check that the NVIDIA driver and CUDA versions are compatible
2. Insufficient memory: adjust batch size or use the CPU version
3. Dependency conflicts: use a virtual environment and install strictly according to the requirements file

## Contribution Guide

Issues and Pull Requests are welcome to help improve the project.

[![Star History Trend](https://api.star-history.com/svg?repos=huangjia2019/langchain-in-action,huangjia2019/ai-agents,huangjia2019/let-us-machine-learning,huangjia2019/rag-in-action,huangjia2019/llm-gpt&type=Date)](https://www.star-history.com/#huangjia2019/langchain-in-action&huangjia2019/ai-agents&huangjia2019/let-us-machine-learning&huangjia2019/rag-in-action&huangjia2019/llm-gpt&Date)

## License

This project is licensed under the MIT License.
"# rag-in-action-english" 
