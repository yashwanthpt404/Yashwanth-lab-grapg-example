# 📘 Retrieval-Augmented Generation (RAG) Application

A comprehensive demonstration of building a Retrieval-Augmented Generation (RAG) pipeline using LangChain, LangGraph for agent workflows, and LangSmith for tracing and evaluation. This project is designed for educational purposes and team demonstrations.

## 🎯 Project Overview

This Jupyter Notebook demonstrates:
- **LangChain**: Building a RAG pipeline with tools and memory
- **LangGraph**: Modeling agent workflows with retry and fallback mechanisms
- **LangSmith**: Tracing, debugging, and evaluating agent runs
- **Ollama**: Using local LLM (Llama 3.1) for inference
- **Chroma**: Vector database for document storage and retrieval

## 📋 Features

- End-to-end RAG pipeline with HR document processing
- Interactive document Q&A system
- Agent workflow visualization
- Comprehensive tracing and debugging
- Memory management for conversational context
- Error handling and fallback strategies

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   Vector Store   │    │   LLM (Ollama)  │
│   (PDF/Text)    │───▶│   (ChromaDB)     │───▶│   Llama 3.1     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Text Splitting  │    │ Semantic Search  │    │ Response Gen    │
│ & Chunking      │    │ & Retrieval      │    │ with Context    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠 Technology Stack

- **LLM**: Llama 3.1 (via Ollama)
- **Vector Store**: Chroma
- **Embeddings**: HuggingFace Embeddings
- **Framework**: LangChain + LangGraph
- **Monitoring**: LangSmith
- **Document Processing**: PyPDF + Recursive Text Splitter

## 📦 Installation

### Prerequisites

- Python 3.8+
- Git
- 8GB+ RAM (for local LLM)
- Internet connection (for initial setup)

### 1. Repository Setup

```bash
git clone <your-repo-url>
cd rag-application
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv rag_env

# Activate environment
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install langchain langchain-community langchain-ollama
pip install langchain-huggingface langgraph langsmith
pip install chromadb pypdf python-dotenv jupyter
```

### 4. Ollama Installation

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

```bash
# Pull the Llama 3.1 model
ollama pull llama3.1

# Verify installation
ollama list
```

## 🔧 Configuration

### 1. LangSmith Setup

#### Get API Key:
1. Visit [LangSmith](https://smith.langchain.com/)
2. Create a free account
3. Navigate to Settings → API Keys
4. Generate new API key

#### Environment Configuration:
Create a `.env` file in the project root:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="RAG-Demo"
```

⚠️ **Important**: Add `.env` to your `.gitignore` file

### 2. Project Structure

*Project structure will be updated based on the provided screenshot*

```
rag-application/
├── README.md
├── .env
├── RAG.ipynb               # Main notebook
├── HR_doc.pdf             # Sample HR document
```

## 🚀 Quick Start

### 1. Start Services

```bash
# Start Ollama service
ollama serve
```

### 2. Launch Application

```bash
# Start Jupyter Notebook
jupyter notebook

# Open rag_demo.ipynb and run cells sequentially
```

### 3. Test the System

Try these sample queries:
- "What is the company's vacation policy?"
- "How do I submit a timesheet?"
- "What are the performance review criteria?"

## 📚 Workflow Breakdown

### 1. Document Ingestion
- **Load Documents**: Extract text from PDF/Markdown files using PyPDF
- **Text Chunking**: Split content using RecursiveCharacterTextSplitter
- **Metadata Handling**: Preserve document structure and context

### 2. Embedding Generation
- **Vectorization**: Generate dense embeddings using HuggingFace models
- **Storage**: Index vectors in ChromaDB with metadata
- **Optimization**: Efficient similarity search capabilities

### 3. Semantic Retrieval
- **Query Processing**: Convert user queries to embeddings
- **Similarity Search**: Retrieve top-k relevant document chunks
- **Context Assembly**: Prepare retrieved content for LLM

### 4. Response Generation
- **Prompt Engineering**: Construct context-aware prompts
- **LLM Inference**: Generate responses using local Llama 3.1
- **Post-processing**: Format and validate outputs

### 5. Agent Workflows (LangGraph)
- **State Management**: Track conversation context
- **Decision Trees**: Handle complex query routing
- **Error Handling**: Implement retry and fallback mechanisms


## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Error:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

**Memory Issues:**
```python
# Reduce chunk size
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Reduced from 1024
    chunk_overlap=50
)
```

**ChromaDB Corruption:**
```python
# Reset database
import shutil
shutil.rmtree("./chroma_db", ignore_errors=True)
```

**LangSmith Tracing Issues:**
- Verify API key in `.env`
- Check internet connectivity
- Confirm `LANGCHAIN_TRACING_V2=true`

## 🔒 Security & Best Practices

### Data Security
- Store sensitive documents securely
- Implement access controls for production
- Regular security audits

### API Management
- Rotate API keys regularly
- Use environment variables for secrets
- Implement rate limiting

### Production Considerations
- Monitor resource usage
- Implement logging and alerting
- Regular model updates and retraining

## 🧪 Testing

### Unit Tests
```bash
# Run tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_rag_pipeline.py -v
```

### Integration Tests
```bash
# Test end-to-end pipeline
python tests/integration/test_full_pipeline.py
```

## 📈 Monitoring & Evaluation

### LangSmith Metrics
- Response latency
- Retrieval accuracy
- User satisfaction scores
- Cost per query

### Custom Metrics
- Document coverage
- Query success rate
- Context relevance scores

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation accordingly


## 🙋‍♂️ Support

For questions or issues:

1. **Documentation**: Check this README and inline comments
2. **Issues**: Create a GitHub issue with detailed description
3. **Discussions**: Use GitHub Discussions for general questions
4. **Community**: Join our Discord/Slack channel

## 🔗 Resources

### Documentation
- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Tutorials
- [RAG Best Practices](https://docs.langchain.com/docs/use-cases/question-answering)
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/)
- [LLM Fine-tuning](https://huggingface.co/docs/transformers/training)

### Community
- [LangChain Community](https://github.com/langchain-ai/langchain)
- [RAG Papers](https://arxiv.org/abs/2005.11401)
- [Best Practices Blog](https://blog.langchain.dev/)

---



