"""
YTRAG v2 - YouTube Transcript RAG System
Mini ChatGPT for YouTube transcripts with visible retrieval + reasoning
"""

import os
from flask import Flask, render_template, request, jsonify
from rag_system_v2 import RAGPipeline

app = Flask(__name__)

# Initialize RAG pipeline
print("Initializing RAG System v2...")
pipeline = RAGPipeline()
pipeline.load_and_index()
print("RAG System v2 ready!")


@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('chat_v2.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Process a question through the RAG pipeline.
    
    Returns:
        {
            "answer": str,
            "confidence": str,
            "refused": bool,
            "reason": str (if refused),
            "sources": List[Dict],
            "chunks": List[Dict],
            "retrieval_scores": List[float],
            "chunks_used": int,
            "query_language": str
        }
    """
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({
            'error': 'Question is required',
            'answer': 'Please provide a question.',
            'confidence': 'low',
            'refused': True,
            'reason': 'Empty question'
        }), 400
    
    try:
        query_lang = pipeline.multilingual.detect_language(question)
        print(f"Detected query language: {query_lang}")
        
        translated_query = None
        if query_lang == "en":
            # Try Hindi translation for Hindi content
            translated_query = pipeline.multilingual.translate_query(question, "hi")
        
        chunks_with_scores = pipeline.retriever.retrieve(question, top_k=5)
        
        # If we have a translated query, also search with it and merge results
        if translated_query and translated_query != question:
            translated_results = pipeline.retriever.retrieve(translated_query, top_k=5)
            # Merge and deduplicate
            seen_texts = set()
            merged = []
            for chunk, score in chunks_with_scores + translated_results:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    merged.append((chunk, score))
            # Sort by score and take top-k
            chunks_with_scores = sorted(merged, key=lambda x: x[1], reverse=True)[:5]
        
        print(f"Retrieved {len(chunks_with_scores)} chunks with scores: {[f'{s:.2f}' for _, s in chunks_with_scores]}")
        
        result = pipeline.generator.generate(question, chunks_with_scores)
        
        # Add retrieval info
        result["retrieval_scores"] = [score for _, score in chunks_with_scores]
        result["chunks_used"] = len(chunks_with_scores)
        result["query_language"] = query_lang
        
        # Prepare response
        response = {
            'answer': result['answer'],
            'confidence': result['confidence'],
            'refused': result['status'] == 'REFUSED',
            'retrieval_scores': result.get('retrieval_scores', []),
            'chunks_used': result.get('chunks_used', 0),
            'query_language': result.get('query_language', 'en')
        }
        
        # Add refusal reason if applicable
        if result['status'] == 'REFUSED':
            response['reason'] = result.get('reason', 'Insufficient context to answer')
        
        # Prepare sources with video links
        sources = []
        if 'sources' in result:
            for source_name in result['sources']:
                # Find video_id for this source
                video_id = None
                for vid, name in pipeline.video_names.items():
                    if name == source_name:
                        video_id = vid
                        break
                
                if video_id:
                    sources.append({
                        'video_name': source_name,
                        'video_url': f'https://www.youtube.com/watch?v={video_id}',
                        'video_id': video_id
                    })
        
        response['sources'] = sources
        
        # Prepare supporting chunks
        chunks = []
        for chunk, score in chunks_with_scores[:3]:
            chunks.append({
                'text': chunk.text,
                'video_name': chunk.video_name,
                'video_id': chunk.video_id,
                'score': round(score, 3),
                'chunk_index': chunk.chunk_index
            })
        
        response['chunks'] = chunks
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({
            'answer': f'An error occurred while processing your question: {str(e)}',
            'confidence': 'low',
            'refused': True,
            'reason': 'System error',
            'sources': [],
            'chunks': [],
            'retrieval_scores': [],
            'chunks_used': 0
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'system': 'RAG v2',
        'chunks_loaded': len(pipeline.chunks),
        'videos': list(pipeline.video_names.values())
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
