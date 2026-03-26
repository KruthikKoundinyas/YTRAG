import os
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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
            'method': 'keyword_matching',
            'score': best_score
        })
    
    return jsonify({'error': 'No matching chunk found'}), 404

@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    """Strict RAG answer generator - detects question type for formatting."""
    data = request.json
    question = data.get('question', '')
    retrieved_chunk = data.get('retrieved_chunk', '')
    
    if not question or not retrieved_chunk:
        return jsonify({'error': 'Question and retrieved chunk required'}), 400
    
    question_lower = question.lower()
    
    # Detect question type
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
        'validation': validation
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)