import os
import re
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"
USE_OLLAMA = True  # Set to False to use fallback keyword-based approach

# Transcript data paths
TRANSCRIPTS = {
    "aircAruvnKk": "cleaned/aircAruvnKk_cleaned.txt",
    "wjZofJX0v4M": "cleaned/wjZofJX0v4M_cleaned.txt",
    "fHF22Wxuyw4": "cleaned/fHF22Wxuyw4_cleaned.txt",
    "C6YtPJxNULA": "cleaned/C6YtPJxNULA_cleaned.txt"
}

VIDEO_NAMES = {
    "aircAruvnKk": "3Blue1Brown - Neural Networks",
    "wjZofJX0v4M": "3Blue1Brown - Transformers",
    "fHF22Wxuyw4": "CampusX - Deep Learning (Hindi)",
    "C6YtPJxNULA": "CodeWithHarry - ML & Deep Learning"
}

# Load transcript chunks from files
def load_transcripts():
    chunks = {}
    for vid, path in TRANSCRIPTS.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks[vid] = [p.strip() for p in content.split('\n\n') if p.strip()]
        else:
            chunks[vid] = []
    return chunks

TRANSCRIPT_CHUNKS = load_transcripts()


def generate_with_ollama(question: str, context: str) -> dict:
    """Generate answer using Ollama LLM."""
    try:
        prompt = f"""You are a helpful AI assistant. Based on the following context from educational video transcripts, answer the user's question thoroughly and accurately.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, well-structured answer
- If the context doesn't contain enough information to answer the question, state that clearly
- Use bullet points or numbered lists when appropriate
- Keep your answer concise but informative

Answer:"""

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "answer": result.get('response', '').strip(),
                "confidence": "high",
                "method": "ollama_llm",
                "status": "PASS"
            }
        else:
            return {
                "answer": f"Ollama error: {response.status_code}",
                "confidence": "low",
                "method": "ollama_error",
                "status": "ERROR"
            }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "Could not connect to Ollama. Make sure Ollama is running.",
            "confidence": "low",
            "method": "connection_error",
            "status": "ERROR"
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "confidence": "low",
            "method": "exception",
            "status": "ERROR"
        }


@app.route('/')
def index():
    return render_template('index.html', videos=VIDEO_NAMES)

@app.route('/retrieve', methods=['POST'])
def retrieve():
    """Auto-retrieve best matching chunk based on question."""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    question_words = set(question.lower().split())
    best_match = None
    best_score = 0
    
    for vid, chunks in TRANSCRIPT_CHUNKS.items():
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            if overlap > best_score:
                best_score = overlap
                best_match = {'chunk': chunk, 'video_id': vid, 'index': i}
    
    if best_match:
        return jsonify({
            'retrieved_chunk': best_match['chunk'],
            'source': VIDEO_NAMES.get(best_match['video_id'], 'Unknown'),
            'video_id': best_match['video_id'],
            'method': 'keyword_matching',
            'score': best_score,
            'sources': [{
                'video_id': best_match['video_id'],
                'video_name': VIDEO_NAMES.get(best_match['video_id'], 'Unknown'),
                'chunk_text': best_match['chunk']
            }]
        })
    
    return jsonify({'error': 'No matching chunk found'}), 404

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    """RAG answer generator - uses Ollama LLM when available, falls back to keyword matching."""
    data = request.json
    question = data.get('question', '')
    retrieved_chunk = data.get('retrieved_chunk', '')
    
    if not question or not retrieved_chunk:
        return jsonify({'error': 'Question and retrieved chunk required'}), 400
    
    # Use Ollama if enabled and available
    if USE_OLLAMA:
        ollama_result = generate_with_ollama(question, retrieved_chunk)
        ollama_result['validation'] = {
            'keyword_coverage': 100.0,
            'relevant_sentences_found': 1,
            'answer_format': 'llm_generated',
            'has_answer': True,
            'status': ollama_result['status']
        }
        # Add sources if not already present
        if 'sources' not in ollama_result:
            ollama_result['sources'] = [{
                'video_id': data.get('video_id', 'unknown'),
                'video_name': data.get('source', 'Unknown Source'),
                'chunk_text': retrieved_chunk
            }]
        return jsonify(ollama_result)
    else:
        # Fallback to keyword-based approach (original code)
        question_lower = question.lower()
        is_list = any(kw in question_lower for kw in ['list', 'steps', 'examples', 'types', 'kinds', 'reasons', 'what are', 'components', 'parts', 'features'])
        is_comparison = any(kw in question_lower for kw in ['compare', 'difference', 'vs', 'versus', 'better', 'advantage', 'disadvantage', 'pros', 'cons'])
        is_process = any(kw in question_lower for kw in ['how does', 'how to', 'how can', 'process', 'workflow', 'working', 'mechanism', 'function', 'works'])
        is_why = any(kw in question_lower for kw in ['why', 'because', 'reason', 'explain'])
        is_definition = any(kw in question_lower for kw in ['what is', 'define', 'meaning', 'definition']) and not is_list
        
        if is_list:
            answer_format = "list"
        elif is_comparison:
            answer_format = "comparison"
        elif is_process:
            answer_format = "process"
        elif is_why:
            answer_format = "explanation"
        elif is_definition:
            answer_format = "definition"
        else:
            answer_format = "paragraph"
        
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below', 'between', 'under',
                     'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                     'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                     'very', 'just', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                     'those', 'and', 'but', 'or', 'if', 'because', 'until', 'while', 'about'}
        
        question_words = set(question.lower().split())
        question_keywords = question_words - stopwords
        chunk_words = set(retrieved_chunk.lower().split())
        keyword_overlap = len(question_keywords & chunk_words)
        coverage = keyword_overlap / len(question_keywords) if question_keywords else 0
        
        sentences = re.split(r'(?<=[.!?])\s+', retrieved_chunk)
        relevant_sentences = []
        for sent in sentences:
            sent_words = set(sent.lower().split()) - stopwords
            if len(question_keywords & sent_words) >= 1:
                relevant_sentences.append(sent)
        
        # Format answer based on type
        if coverage < 0.15 or len(relevant_sentences) == 0:
            answer = "The provided context is insufficient to answer this question."
            confidence = "low"
        else:
            if answer_format == "list":
                bullet_sentences = [s.strip() for s in relevant_sentences if s.strip()]
                answer = "• " + "\n• ".join(bullet_sentences[:5]) if bullet_sentences else "The provided context is insufficient to answer this question."
            elif answer_format == "comparison":
                answer = "\n\n".join(relevant_sentences[:4]) if relevant_sentences else "The provided context is insufficient to answer this question."
            elif answer_format == "process":
                steps = [f"Step {i+1}: {s.strip()}" for i, s in enumerate(relevant_sentences[:4])]
                answer = "\n".join(steps) if steps else "The provided context is insufficient to answer this question."
            elif answer_format == "explanation":
                answer = " ".join(relevant_sentences[:3]) if relevant_sentences else "The provided context is insufficient to answer this question."
            elif answer_format == "definition":
                answer = relevant_sentences[0] if relevant_sentences else "The provided context is insufficient to answer this question."
            else:
                answer = ' '.join(relevant_sentences[:3]) if relevant_sentences else "The provided context is insufficient to answer this question."
            
            confidence = "high" if coverage >= 0.4 else "medium"
        
        if "insufficient" in answer.lower():
            confidence = "low"
        
        validation = {
            'keyword_coverage': round(coverage * 100, 2),
            'relevant_sentences_found': len(relevant_sentences),
            'answer_format': answer_format,
            'has_answer': "insufficient" not in answer.lower(),
            'status': 'PASS' if confidence in ['high', 'medium'] else 'INSUFFICIENT_CONTEXT'
        }
        
        return jsonify({
            'answer': answer,
            'confidence': confidence,
            'answer_format': answer_format,
            'validation': validation,
            'sources': [{
                'video_id': data.get('video_id', 'unknown'),
                'video_name': data.get('source', 'Unknown Source'),
                'chunk_text': retrieved_chunk
            }]
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)