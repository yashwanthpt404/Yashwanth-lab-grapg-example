# 📘 Retrieval-Augmented Generation (RAG) Notebook

This project demonstrates the implementation and working of Retrieval-Augmented Generation (RAG), an advanced method that enhances the capabilities of language models by integrating external knowledge retrieved from documents during response generation. This approach improves factual accuracy and domain specificity, making it ideal for enterprise use cases, such as customer service, technical support, and internal knowledge base assistants.

## 🚀 Project Overview

- **Notebook Name**: `RAG.ipynb`
- **Objective**: To showcase the integration of vector databases with a retrieval-augmented generation (RAG) pipeline using open-source LLMs.
- **Scope**: Indexing a document, performing semantic search, and generating relevant responses using a local language model.

---

## 🧰 Technologies Used

| Component              | Description                                          |
|------------------------|------------------------------------------------------|
| 🧠 LLM                 | `llama3.1` or equivalent via `Ollama`                  |
| 📚 Vector Store        | `ChromaDB`                                           |
| 🛠️ Embeddings         | `sentence-transformers` or `ollama` embeddings       |
| 💻 Interface           | Python via Jupyter Notebook                          |
| 🔄 Retrieval Pipeline  | Custom semantic search and chunking mechanism        |

---

## 🗂️ Workflow Breakdown

### 1. **Document Ingestion**
- Load PDF using `PyMuPDF` (`fitz`) to extract raw text content.
- Split content into manageable text chunks using `LangChain`'s `RecursiveCharacterTextSplitter`.

### 2. **Embedding Generation**
- Generate dense vector embeddings for each chunk using `SentenceTransformer` or `Ollama`'s embedding model.
- Store vectors in `ChromaDB` for efficient similarity search.

### 3. **Semantic Retrieval**
- On receiving a query, perform a similarity search over the vector store.
- Retrieve top-k relevant chunks as context for the query.

### 4. **Response Generation**
- Use `Ollama` to run a local LLM (e.g., LLaMA 3.1).
- Construct a prompt with the retrieved context + query.
- Generate a grounded, context-aware response.

---

## 📌 Key Highlights

- ✅ **Self-hosted RAG** pipeline without reliance on cloud APIs.
- ✅ **Semantic Search** enabled through sentence embeddings and vector DB.
- ✅ **Dynamic Prompt Engineering**: Injecting retrieved text directly into the model prompt.
- ✅ **Reproducibility**: Designed to work locally using `ollama run`.

---
