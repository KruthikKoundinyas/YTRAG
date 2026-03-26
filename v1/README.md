# YTRAG v1 - YouTube Transcript RAG (Basic Version)

A simple RAG system for querying YouTube video transcripts using keyword matching and Ollama LLM.

## 🎯 Overview

YTRAG v1 is the original version that provides:
- **Keyword-based retrieval** using word overlap
- **Ollama integration** for AI-powered answer generation
- **Fallback mode** that works without Ollama
- **Simple chat interface** for asking questions

## 📁 Files

```
v1/
├── app.py           # Flask application with routes
├── index.html       # Main page template
└── chat.html        # Chat interface template
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Flask
- Requests
- Ollama (optional, for LLM-powered answers)

### Installation

```bash
pip install flask requests
```

### Running the Application

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Without Ollama

The system works without Ollama using a fallback keyword-based approach. To disable Ollama:

1. Open `app.py`
2. Set `USE_OLLAMA = False`
3. Restart the application

## 🎥 Available Videos

The system includes transcripts from:
- **3Blue1Brown - Neural Networks** (English)
- **3Blue1Brown - Transformers** (English)
- **CampusX - Deep Learning** (Hindi)
- **CodeWithHarry - ML & Deep Learning** (English)

## 🔧 How It Works

### Retrieval Process

1. **User asks a question** via the chat interface
2. **Keyword matching** finds the best matching transcript chunk
3. **Answer generation** uses Ollama LLM or keyword extraction
4. **Response** includes answer, confidence, and sources

### Answer Generation

#### With Ollama (Default)
- Sends question + context to Ollama LLM
- Uses Llama3 model for answer generation
- Low temperature (0.3) for factual answers
- Returns structured response with confidence

#### Without Ollama (Fallback)
- Extracts relevant sentences from context
- Formats answer based on question type:
  - **List questions**: Bullet points
  - **Process questions**: Numbered steps
  - **Comparison questions**: Side-by-side
  - **Definition questions**: Single sentence
- Calculates confidence based on keyword coverage

## 📊 API Endpoints

### `GET /`
Returns the main chat interface.

### `POST /retrieve`
Retrieves the best matching chunk for a question.

**Request:**
```json
{
  "question": "What is a neural network?"
}
```

**Response:**
```json
{
  "retrieved_chunk": "A neural network is...",
  "source": "3Blue1Brown - Neural Networks",
  "video_id": "aircAruvnKk",
  "method": "keyword_matching",
  "score": 5,
  "sources": [...]
}
```

### `POST /generate_answer`
Generates an answer from a retrieved chunk.

**Request:**
```json
{
  "question": "What is a neural network?",
  "retrieved_chunk": "A neural network is...",
  "video_id": "aircAruvnKk",
  "source": "3Blue1Brown - Neural Networks"
}
```

**Response:**
```json
{
  "answer": "A neural network is...",
  "confidence": "high",
  "answer_format": "paragraph",
  "validation": {...},
  "sources": [...]
}
```

## 🎨 UI Features

- **Chat interface** with message history
- **Example questions** for quick testing
- **Loading animation** during processing
- **Confidence indicators** (HIGH/MEDIUM/LOW)
- **Source attribution** with video links
- **Responsive design** for mobile/desktop

## 📝 Example Questions

- "What is a neural network?"
- "How do transformers work?"
- "Explain backpropagation"
- "What is gradient descent?"
- "Why are biases necessary in neural networks?"

## ⚙️ Configuration

### Ollama Settings

Edit `app.py` to configure Ollama:

```python
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama API URL
OLLAMA_MODEL = "llama3"                      # Model name
USE_OLLAMA = True                            # Enable/disable Ollama
```

### Transcript Paths

Edit `app.py` to add/remove videos:

```python
TRANSCRIPTS = {
    "video_id": "cleaned/video_id_cleaned.txt",
    ...
}

VIDEO_NAMES = {
    "video_id": "Video Title",
    ...
}
```

## 🔍 Limitations

- **Keyword matching**: May miss semantically similar content
- **Single chunk**: Only uses one chunk for answer generation
- **No multilingual**: Limited cross-language support
- **No grounding enforcement**: May hallucinate if context insufficient

## 🚀 Upgrading to v2

For better performance and features, upgrade to v2:
- Semantic retrieval using embeddings
- FAISS index for fast search
- Strict grounding enforcement
- Hallucination detection
- Multilingual support

See [`../v2/README.md`](../v2/README.md) for details.

## 🛠️ Troubleshooting

### Ollama Connection Error

```
Could not connect to Ollama. Make sure Ollama is running.
```

**Solution:**
1. Install Ollama: https://ollama.ai
2. Pull the model: `ollama pull llama3`
3. Start Ollama: `ollama serve`

### No Matching Chunk Found

```
No matching chunk found
```

**Solution:**
- Try different question phrasing
- Check if transcripts are loaded
- Verify cleaned/ directory has transcript files

### Low Confidence Answers

**Solution:**
- Use more specific questions
- Include keywords from the video topic
- Try rephrasing the question

## 📚 Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [3Blue1Brown YouTube](https://www.youtube.com/@3blue1brown)
- [CampusX YouTube](https://www.youtube.com/@CampusX-official)

## 📄 License

This project is open source and available under the MIT License.
