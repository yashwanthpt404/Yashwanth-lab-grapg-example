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

# RAG Application with LangChain, LangGraph & LangSmith

A comprehensive demonstration of building a Retrieval-Augmented Generation (RAG) pipeline using LangChain, LangGraph for agent workflows, and LangSmith for tracing and evaluation. This project is designed for educational purposes and team demonstrations.

## 🎯 Project Overview

This Jupyter Notebook demonstrates:
- *LangChain*: Building a RAG pipeline with tools and memory
- *LangGraph*: Modeling agent workflows with retry and fallback mechanisms
- *LangSmith*: Tracing, debugging, and evaluating agent runs
- *Ollama*: Using local LLM (Llama 3.1) for inference
- *Chroma*: Vector database for document storage and retrieval

## 📋 Features

- End-to-end RAG pipeline with HR document processing
- Interactive document Q&A system
- Agent workflow visualization
- Comprehensive tracing and debugging
- Memory management for conversational context
- Error handling and fallback strategies

## 🛠 Technology Stack

- *LLM*: Llama 3.1 (via Ollama)
- *Vector Store*: Chroma
- *Embeddings*: HuggingFace Embeddings
- *Framework*: LangChain + LangGraph
- *Monitoring*: LangSmith
- *Document Processing*: PyPDF + Recursive Text Splitter

## 📦 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Git

### 1. Clone the Repository
bash
git clone <your-repo-url>
cd rag-application


### 2. Create Virtual Environment
bash
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate


### 3. Install Dependencies
bash
pip install langchain
pip install langchain-community
pip install langchain-ollama
pip install langchain-huggingface
pip install langgraph
pip install langsmith
pip install chromadb
pip install pypdf
pip install python-dotenv
pip install jupyter


Or install from requirements file:
bash
pip install -r requirements.txt


### 4. Install Ollama
Download and install Ollama from [https://ollama.ai](https://ollama.ai)

Pull the Llama 3.1 model:
bash
ollama pull llama3.1


Verify installation:
bash
ollama list


## 🔧 Configuration

### 1. LangSmith Setup

#### Get Your LangSmith API Key:
1. Visit [LangSmith](https://smith.langchain.com/)
2. Sign up for a free account
3. Navigate to Settings → API Keys
4. Create a new API key
5. Copy the API key

#### Create Environment File:
Create a .env file in the project root:
env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="RAG-Demo"


⚠ *Important*: Never commit your .env file to version control!

### 2. Project Structure

rag-application/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── rag_demo.ipynb
├── documents/
│   └── sample_hr_document.pdf
└── chroma_db/
    └── (vector store files)


## 🚀 Quick Start

### 1. Start Ollama Service
bash
ollama serve


### 2. Launch Jupyter Notebook
bash
jupyter notebook


### 3. Open the Demo Notebook
Navigate to rag_demo.ipynb and run the cells sequentially.

## 📚 Notebook Walkthrough

### Section 1: Environment Setup
- Import necessary libraries
- Load environment variables
- Initialize LangSmith tracing

### Section 2: Document Processing
- Load HR documents (PDF/Markdown)
- Text splitting and chunking
- Generate embeddings
- Store in Chroma vector database

### Section 3: RAG Pipeline
- Initialize Llama 3.1 via Ollama
- Create retrieval chain
- Implement Q&A functionality

### Section 4: LangGraph Workflows
- Design agent workflows
- Implement retry mechanisms
- Add fallback strategies
- Visualize workflow graphs

### Section 5: Memory & Tools
- Conversational memory setup
- Custom tool integration
- Multi-turn conversations

### Section 6: LangSmith Integration
- Trace logging and monitoring
- Performance evaluation
- Debugging workflows

## 🔍 Key Concepts Explained

### RAG (Retrieval-Augmented Generation)
RAG combines the power of large language models with external knowledge retrieval, enabling more accurate and contextually relevant responses.

### LangGraph Workflows
State-based agent workflows that can handle complex decision trees, retries, and error recovery patterns.

### Vector Embeddings
Numerical representations of text that enable semantic similarity search and efficient document retrieval.

## 📊 Sample Queries

Try these example queries in the notebook:
- "What is the company's vacation policy?"
- "How do I submit a timesheet?"
- "What are the performance review criteria?"
- "Tell me about the benefits package"

## 🐛 Troubleshooting

### Common Issues:

*Ollama Connection Error:*
bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list


*LangSmith Tracing Not Working:*
- Verify API key in .env file
- Check internet connection
- Ensure LANGCHAIN_TRACING_V2=true

*Chroma Database Issues:*
python
# Reset Chroma database if corrupted
import shutil
shutil.rmtree("./chroma_db", ignore_errors=True)


*Memory Issues with Large Documents:*
- Reduce chunk size in text splitter
- Use smaller embedding models
- Process documents in batches

## 📈 Performance Optimization

- *Chunk Size*: Experiment with different text chunk sizes (512, 1024, 2048 tokens)
- *Overlap*: Use 10-20% overlap between chunks
- *Top-K Retrieval*: Start with k=3-5 for most queries
- *Embedding Model*: Balance between speed and accuracy

## 🔒 Security Best Practices

- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate API keys
- Implement rate limiting for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂ Support

For questions or issues:
1. Check the troubleshooting section
2. Review LangChain documentation
3. Create an issue in this repository
4. Contact the development team

## 🔗 Useful Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Chroma Documentation](https://docs.trychroma.com/)

---
