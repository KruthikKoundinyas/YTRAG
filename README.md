# YTRAG - YouTube Transcript RAG System

A Retrieval-Augmented Generation (RAG) system for querying YouTube video transcripts with AI-powered answers.

## 🎯 Overview

YTRAG allows you to ask questions about YouTube video transcripts and get intelligent answers with:
- **Semantic search** using multilingual embeddings
- **Grounded answers** that only use information from the transcripts
- **Source attribution** with clickable YouTube links
- **Confidence scoring** and refusal detection
- **Multilingual support** (English/Hindi)

## 📁 Project Structure

```
YTRAG/
├── v1/                          # Version 1 - Basic keyword matching
│   ├── app.py                   # Flask application
│   ├── index.html               # Main page template
│   └── chat.html                # Chat interface template
│
├── v2/                          # Version 2 - Production-ready RAG
│   ├── app_v2.py                # Flask application with RAG pipeline
│   ├── rag_system_v2.py         # Core RAG system (semantic retrieval, grounding, multilingual)
│   ├── chat_v2.html             # Enhanced ChatGPT-like UI
│   ├── requirements_v2.txt      # Python dependencies
│   └── RAG_SYSTEM_V2_GUIDE.md   # Detailed technical guide
│
├── tests/                       # Utility scripts
│   ├── fetch_transcript.py      # Fetch single transcript
│   ├── fetch_all.py             # Fetch all transcripts
│   └── clean_transcripts.py     # Clean and preprocess transcripts
│
├── cleaned/                     # Cleaned transcript files
├── temp_transcripts/            # Raw transcript files
└── README.md                    # This file
```

## 🚀 Quick Start

### Version 2 (Recommended)

The production-ready version with semantic retrieval and strict grounding.

```bash
cd v2
pip install -r requirements_v2.txt
python app_v2.py
```

Open http://localhost:5001 in your browser.

### Version 1

The original version with keyword-based matching.

```bash
cd v1
pip install flask requests
python app.py
```

Open http://localhost:5000 in your browser.

## 🎥 Available Videos

The system includes transcripts from:
- **3Blue1Brown - Neural Networks** (English)
- **3Blue1Brown - Transformers** (English)
- **CampusX - Deep Learning** (Hindi)
- **CodeWithHarry - ML & Deep Learning** (English)

## 🔧 Features

### Version 2 (RAG System)

- **Semantic Retrieval**: Uses `multilingual-e5-large` embeddings for cross-lingual search
- **FAISS Index**: Fast approximate nearest neighbor search
- **Strict Grounding**: Only answers from provided context, refuses otherwise
- **Hallucination Detection**: Detects when LLM adds external knowledge
- **Multilingual Support**: Handles English and Hindi queries
- **Confidence Scoring**: HIGH/MEDIUM/LOW based on retrieval scores
- **Visible Retrieval**: Shows supporting chunks, sources, and scores

### Version 1 (Basic)

- **Keyword Matching**: Simple word overlap for retrieval
- **Ollama Integration**: Uses local LLM for answer generation
- **Fallback Mode**: Works without Ollama using keyword extraction
- **Basic UI**: Simple chat interface

## 📊 How It Works

### RAG Pipeline (v2)

```
User Query
    ↓
Language Detection (en/hi)
    ↓
Semantic Retrieval (FAISS + multilingual-e5-large)
    ↓
Re-ranking (score-based)
    ↓
Grounded Answer Generation (Ollama LLM)
    ↓
Validation (hallucination detection, refusal check)
    ↓
Response (answer + sources + chunks + confidence)
```

### Key Components

1. **SemanticRetriever**: Encodes queries and chunks using multilingual embeddings, searches FAISS index
2. **GroundedAnswerGenerator**: Generates answers using Ollama with strict grounding prompts
3. **MultilingualHandler**: Detects query language and handles cross-lingual retrieval
4. **FailureGuardrails**: Rules for refusal, hallucination detection, and bad retrieval detection

## 🛠️ Adding New Videos

1. Fetch transcript:
```bash
cd tests
python fetch_transcript.py <VIDEO_ID>
```

2. Clean transcript:
```bash
python clean_transcripts.py
```

3. Restart the app to rebuild the index.

## 📝 Example Questions

- "What is a neural network?"
- "How do transformers work?"
- "Explain backpropagation"
- "What is gradient descent?"
- "Why are biases necessary in neural networks?"

## 🔍 Technical Details

### Embedding Model
- **Model**: `intfloat/multilingual-e5-large`
- **Dimension**: 1024
- **Languages**: 100+ languages (English, Hindi, etc.)

### Vector Database
- **FAISS**: Facebook AI Similarity Search
- **Index Type**: IndexFlatIP (Inner Product for cosine similarity)
- **Normalization**: L2 normalized embeddings

### LLM
- **Model**: llama3 (via Ollama)
- **Temperature**: 0.1 (low for factual answers)
- **Max Tokens**: 512

## 📚 Documentation

For detailed technical documentation, see [`v2/RAG_SYSTEM_V2_GUIDE.md`](v2/RAG_SYSTEM_V2_GUIDE.md).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **3Blue1Brown** for excellent educational content
- **CampusX** for Hindi deep learning tutorials
- **CodeWithHarry** for ML & Deep Learning content
- **Sentence Transformers** for multilingual embeddings
- **FAISS** for efficient similarity search
- **Ollama** for local LLM inference
