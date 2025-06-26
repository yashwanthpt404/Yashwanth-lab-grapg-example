# 📘 Retrieval-Augmented Generation (RAG) Application

A comprehensive demonstration of building a production-ready Retrieval-Augmented Generation (RAG) pipeline using LangChain, LangGraph, and LangSmith. This project showcases the integration of vector databases with local LLMs to create intelligent document Q&A systems for enterprise use cases.

## 🎯 Project Overview

This project implements a complete RAG pipeline that enhances language model capabilities by integrating external knowledge retrieved from documents during response generation. The system is designed for enterprise applications such as customer service, technical support, and internal knowledge base assistants.

### Key Features

- **End-to-end RAG pipeline** with document processing and retrieval
- **Local LLM integration** using Ollama (Llama 3.1)
- **Agent workflows** with LangGraph for complex decision trees
- **Comprehensive monitoring** with LangSmith tracing
- **Interactive Q&A system** with conversational memory
- **Self-hosted solution** without cloud API dependencies

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

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Llama 3.1 (via Ollama) | Text generation and reasoning |
| **Vector Store** | ChromaDB | Document embeddings storage |
| **Embeddings** | HuggingFace/Sentence Transformers | Text vectorization |
| **Framework** | LangChain + LangGraph | RAG pipeline orchestration |
| **Monitoring** | LangSmith | Tracing and evaluation |
| **Document Processing** | PyPDF + RecursiveCharacterTextSplitter | Text extraction and chunking |
| **Interface** | Jupyter Notebook | Interactive development |

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

```
rag-application/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── rag_demo.ipynb           # Main notebook
├── documents/               # Sample documents
│   └── sample_hr_document.pdf
├── chroma_db/              # Vector store (auto-generated)
└── src/                    # Additional utilities (optional)
    ├── __init__.py
    ├── document_processor.py
    ├── rag_pipeline.py
    └── utils.py
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

## 🔍 Advanced Features

### Memory Management
```python
# Conversational memory for multi-turn interactions
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # Remember last 5 exchanges
    return_messages=True
)
```

### Custom Tools Integration
```python
# Example custom tool for specific operations
@tool
def calculate_days_between_dates(start_date: str, end_date: str) -> str:
    """Calculate days between two dates."""
    # Implementation here
    pass
```

### Workflow Visualization
```python
# Visualize LangGraph workflows
from langgraph.visualization import visualize_graph
visualize_graph(workflow)
```

## 📊 Performance Optimization

### Chunking Strategy
- **Chunk Size**: 512-1024 tokens for balanced context
- **Overlap**: 20% overlap between chunks
- **Splitting**: Respect document structure (paragraphs, sections)

### Retrieval Tuning
- **Top-K**: Start with k=3-5 relevant chunks
- **Similarity Threshold**: Filter low-relevance results
- **Reranking**: Optional reranking for better precision

### Memory Optimization
- **Model Quantization**: Use 4-bit/8-bit quantized models
- **Batch Processing**: Process documents in batches
- **Caching**: Cache embeddings and frequent queries

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

## 📋 Requirements.txt

```txt
langchain>=0.1.0
langchain-community>=0.0.20
langchain-ollama>=0.1.0
langchain-huggingface>=0.0.3
langgraph>=0.0.40
langsmith>=0.1.0
chromadb>=0.4.0
pypdf>=4.0.0
python-dotenv>=1.0.0
jupyter>=1.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0  # Optional: for web interface
gradio>=4.0.0      # Optional: for web interface
```

---
