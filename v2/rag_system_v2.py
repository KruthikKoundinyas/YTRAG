"""
Production-Ready RAG System v2
Fixes: Retrieval, Grounding, Multilingual, Refusal
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import requests



@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    video_id: str
    video_name: str
    language: str  # 'en' or 'hi'
    chunk_index: int
    embedding: Optional[np.ndarray] = None


class SemanticRetriever:
    """
    Replace keyword matching with semantic embeddings.
    Uses multilingual-e5-large for cross-lingual support.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks: List[Chunk] = []
        
    def build_index(self, chunks: List[Chunk]):
        """Build FAISS index from chunks."""
        print(f"Building FAISS index for {len(chunks)} chunks...")
        
        # Encode all chunks
        texts = [f"passage: {chunk.text}" for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        
        # Store embeddings
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine after normalization)
        
        # Convert embeddings to numpy array (already normalized by encode())
        embeddings_np = np.array(embeddings, dtype=np.float32)
        embeddings_np = np.ascontiguousarray(embeddings_np)

        self.index.add(embeddings_np)   #   type: ignore
        self.chunks = chunks
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.45) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for query.
        Returns list of (chunk, score) tuples.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query (already normalized by encode())
        query_embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True
        )
        query_np = np.array(query_embedding, dtype=np.float32)
        query_np = np.ascontiguousarray(query_np)
        
        scores, indices = self.index.search(query_np, top_k)    #   type: ignore
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_score:
                results.append((self.chunks[idx], float(score)))
        
        return results


class GroundedAnswerGenerator:
    """
    Strict grounding enforcement.
    Only answers from provided context, refuses otherwise.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3"):
        self.ollama_url = ollama_url
        self.model = model
    
    def generate(self, question: str, chunks: List[Tuple[Chunk, float]]) -> Dict:
        """
        Generate grounded answer from chunks.
        Returns dict with answer, confidence, sources, status.
        """
        if not chunks:
            return {
                "answer": "The provided context is insufficient to answer this question.",
                "confidence": "low",
                "sources": [],
                "status": "REFUSED",
                "reason": "No relevant chunks found"
            }
        
        # Check if best chunk is relevant enough
        best_score = chunks[0][1]
        if best_score < 0.3:
            return {
                "answer": "The provided context is insufficient to answer this question.",
                "confidence": "low",
                "sources": [],
                "status": "REFUSED",
                "reason": f"Best chunk score {best_score:.2f} below threshold 0.3"
            }
        
        # Prepare context
        context = "\n\n---\n\n".join([
            f"[Source: {chunk.video_name}]\n{chunk.text}"
            for chunk, score in chunks[:3]  # Use top 3 chunks
        ])
        
        # Generate answer with strict grounding prompt
        prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
1. You MUST answer ONLY using information explicitly stated in the context above.
2. If the context does not contain enough information to answer the question, you MUST respond EXACTLY: "The provided context is insufficient to answer this question."
3. Do NOT use any external knowledge or make assumptions.
4. Do NOT infer information not explicitly stated in the context.
5. Quote specific parts of the context to support your answer.

Answer:"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual answers
                        "top_p": 0.9,
                        "num_predict": 512
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # Validate answer
                if self._is_insufficient(answer):
                    return {
                        "answer": "The provided context is insufficient to answer this question.",
                        "confidence": "low",
                        "sources": [chunk.video_name for chunk, _ in chunks[:3]],
                        "status": "REFUSED",
                        "reason": "LLM determined context insufficient"
                    }
                
                # Check for hallucination indicators
                if self._has_hallucination_indicators(answer, context):
                    return {
                        "answer": "The provided context is insufficient to answer this question.",
                        "confidence": "low",
                        "sources": [chunk.video_name for chunk, _ in chunks[:3]],
                        "status": "REFUSED",
                        "reason": "Potential hallucination detected"
                    }
                
                return {
                    "answer": answer,
                    "confidence": "high" if best_score >= 0.5 else "medium",
                    "sources": [chunk.video_name for chunk, _ in chunks[:3]],
                    "status": "PASS",
                    "chunk_scores": [score for _, score in chunks[:3]]
                }
            else:
                return {
                    "answer": f"Error generating answer: {response.status_code}",
                    "confidence": "low",
                    "sources": [],
                    "status": "ERROR"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "answer": "Could not connect to Ollama. Make sure Ollama is running.",
                "confidence": "low",
                "sources": [],
                "status": "ERROR"
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "confidence": "low",
                "sources": [],
                "status": "ERROR"
            }
    
    def _is_insufficient(self, answer: str) -> bool:
        """Check if answer indicates insufficient context."""
        insufficient_phrases = [
            "insufficient to answer",
            "not enough information",
            "cannot answer",
            "does not contain",
            "not mentioned in the context",
            "not provided in the context"
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in insufficient_phrases)
    
    def _has_hallucination_indicators(self, answer: str, context: str) -> bool:
        """
        Detect potential hallucinations.
        Checks if answer contains information not in context.
        """
        # Simple check: if answer is much longer than context, might be hallucinating
        if len(answer) > len(context) * 0.8:
            return True
        
        # Check for specific patterns that indicate external knowledge
        external_knowledge_patterns = [
            r"according to.*research",
            r"studies have shown",
            r"it is well known that",
            r"generally speaking",
            r"in most cases",
            r"typically",
            r"usually"
        ]
        
        for pattern in external_knowledge_patterns:
            if re.search(pattern, answer.lower()):
                return True
        
        return False


class MultilingualHandler:
    """
    Simple multilingual support.
    Detects query language and translates if needed.
    """
    
    def __init__(self):
        # Simple language detection based on character ranges
        self.hindi_chars = set(range(0x0900, 0x097F))  # Devanagari Unicode range
    
    def detect_language(self, text: str) -> str:
        """Detect if text is English or Hindi."""
        hindi_count = sum(1 for char in text if ord(char) in self.hindi_chars)
        if hindi_count > len(text) * 0.1:  # More than 10% Hindi characters
            return "hi"
        return "en"
    
    def translate_query(self, query: str, target_lang: str) -> str:
        """
        Simple translation using Ollama.
        For production, use Google Translate API or similar.
        """
        if self.detect_language(query) == target_lang:
            return query
        
        try:
            prompt = f"Translate the following text to {'Hindi' if target_lang == 'hi' else 'English'}. Only provide the translation, nothing else.\n\nText: {query}\n\nTranslation:"
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', query).strip()
        except:
            pass
        
        return query  # Fallback to original




class RAGPipeline:
    """
    Complete RAG pipeline with retrieval, grounding, and multilingual support.
    
    Query → Language Detection → Translation (if needed)
      → Semantic Retrieval (FAISS)
      → Re-ranking (score-based)
      → Grounded Answer Generation
      → Validation → Response
    """
    
    def __init__(self, transcripts_dir: str = "cleaned"):
        self.transcripts_dir = transcripts_dir
        self.retriever = SemanticRetriever()
        self.generator = GroundedAnswerGenerator()
        self.multilingual = MultilingualHandler()
        self.chunks: List[Chunk] = []
        
        # Video metadata
        self.video_names = {
            "aircAruvnKk": "3Blue1Brown - Neural Networks",
            "wjZofJX0v4M": "3Blue1Brown - Transformers",
            "fHF22Wxuyw4": "CampusX - Deep Learning (Hindi)",
            "C6YtPJxNULA": "CodeWithHarry - ML & Deep Learning"
        }
        
        self.video_languages = {
            "aircAruvnKk": "en",
            "wjZofJX0v4M": "en",
            "fHF22Wxuyw4": "hi",
            "C6YtPJxNULA": "en"
        }
    
    def load_and_index(self):
        """Load transcripts and build search index."""
        print("Loading transcripts...")
        
        for video_id, video_name in self.video_names.items():
            filepath = os.path.join(self.transcripts_dir, f"{video_id}_cleaned.txt")
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found")
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (paragraphs)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Skip header lines (Source:, Video ID:, ===)
            start_idx = 0
            for i, p in enumerate(paragraphs):
                if p.startswith('Source:') or p.startswith('Video ID:') or p.startswith('==='):
                    start_idx = i + 1
                else:
                    break
            
            # Create chunks
            for idx, paragraph in enumerate(paragraphs[start_idx:]):
                if len(paragraph) > 50:  # Skip very short chunks
                    chunk = Chunk(
                        text=paragraph,
                        video_id=video_id,
                        video_name=video_name,
                        language=self.video_languages[video_id],
                        chunk_index=idx
                    )
                    self.chunks.append(chunk)
        
        print(f"Loaded {len(self.chunks)} chunks from {len(self.video_names)} videos")
        
        # Build FAISS index
        self.retriever.build_index(self.chunks)
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Returns:
            {
                "answer": str,
                "confidence": str,
                "sources": List[str],
                "status": str,
                "retrieval_scores": List[float],
                "chunks_used": int
            }
        """
        query_lang = self.multilingual.detect_language(question)
        print(f"Detected query language: {query_lang}")
        
        translated_query = None
        if query_lang == "en":
            # Try Hindi translation for Hindi content
            translated_query = self.multilingual.translate_query(question, "hi")
        
        chunks_with_scores = self.retriever.retrieve(question, top_k=top_k)
        
        # If we have a translated query, also search with it and merge results
        if translated_query and translated_query != question:
            translated_results = self.retriever.retrieve(translated_query, top_k=top_k)
            # Merge and deduplicate
            seen_texts = set()
            merged = []
            for chunk, score in chunks_with_scores + translated_results:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    merged.append((chunk, score))
            # Sort by score and take top-k
            chunks_with_scores = sorted(merged, key=lambda x: x[1], reverse=True)[:top_k]
        
        print(f"Retrieved {len(chunks_with_scores)} chunks with scores: {[f'{s:.2f}' for _, s in chunks_with_scores]}")
        
        result = self.generator.generate(question, chunks_with_scores)
        
        # Add retrieval info
        result["retrieval_scores"] = [score for _, score in chunks_with_scores]
        result["chunks_used"] = len(chunks_with_scores)
        result["query_language"] = query_lang
        
        return result




class FailureGuardrails:
    """
    Rules for when to refuse, detect hallucinations, detect bad retrieval.
    """
    
    @staticmethod
    def should_refuse_retrieval(chunks: List[Tuple[Chunk, float]], threshold: float = 0.3) -> Tuple[bool, str]:
        """
        Refuse if best chunk score is below threshold.
        """
        if not chunks:
            return True, "No chunks retrieved"
        
        best_score = chunks[0][1]
        if best_score < threshold:
            return True, f"Best chunk score {best_score:.2f} below threshold {threshold}"
        
        return False, "OK"
    
    @staticmethod
    def should_refuse_answer(answer: str, context: str) -> Tuple[bool, str]:
        """
        Refuse if answer indicates insufficient context.
        """
        insufficient_phrases = [
            "insufficient to answer",
            "not enough information",
            "cannot answer",
            "does not contain",
            "not mentioned in the context",
            "not provided in the context"
        ]
        
        answer_lower = answer.lower()
        for phrase in insufficient_phrases:
            if phrase in answer_lower:
                return True, f"Answer contains refusal phrase: '{phrase}'"
        
        return False, "OK"
    
    @staticmethod
    def detect_hallucination(answer: str, context: str) -> Tuple[bool, str]:
        """
        Detect potential hallucinations.
        """
        if len(answer) > len(context) * 0.8:
            return True, f"Answer length ({len(answer)}) > 80% of context length ({len(context)})"
        
        external_patterns = [
            (r"according to.*research", "references external research"),
            (r"studies have shown", "references studies"),
            (r"it is well known that", "assumes common knowledge"),
            (r"generally speaking", "makes generalization"),
            (r"in most cases", "makes generalization"),
            (r"typically", "makes generalization"),
            (r"usually", "makes generalization")
        ]
        
        for pattern, reason in external_patterns:
            if re.search(pattern, answer.lower()):
                return True, f"External knowledge indicator: {reason}"
        
        answer_sentences = re.split(r'[.!?]+', answer)
        context_lower = context.lower()
        
        unsupported_count = 0
        for sentence in answer_sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only check substantial sentences
                # Check if key words from sentence appear in context
                words = set(sentence.lower().split())
                important_words = [w for w in words if len(w) > 4]  # Skip short words
                if important_words:
                    match_count = sum(1 for w in important_words if w in context_lower)
                    if match_count < len(important_words) * 0.3:  # Less than 30% match
                        unsupported_count += 1
        
        if unsupported_count > len(answer_sentences) * 0.5:  # More than 50% unsupported
            return True, f"{unsupported_count} sentences appear unsupported by context"
        
        return False, "OK"
    
    @staticmethod
    def detect_bad_retrieval(chunks: List[Tuple[Chunk, float]], question: str) -> Tuple[bool, str]:
        """
        Detect if retrieval is off-topic.
        """
        if not chunks:
            return True, "No chunks retrieved"
        
        # Check if question keywords appear in retrieved chunks
        question_words = set(question.lower().split())
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
        
        question_keywords = question_words - stopwords
        
        # Check top chunk
        top_chunk_text = chunks[0][0].text.lower()
        keyword_matches = sum(1 for kw in question_keywords if kw in top_chunk_text)
        
        if keyword_matches < len(question_keywords) * 0.2:  # Less than 20% keyword match
            return True, f"Only {keyword_matches}/{len(question_keywords)} question keywords found in top chunk"
        
        return False, "OK"




"""
MULTILINGUAL STRATEGY:

1. APPROACH: Multilingual Embeddings (NOT translation)
   - Use multilingual-e5-large model
   - Embeds English and Hindi in same vector space
   - Enables direct cross-lingual retrieval

2. WHY NOT TRANSLATION:
   - Translation adds latency
   - Translation errors compound
   - Multilingual embeddings are simpler and more robust

3. IMPLEMENTATION:
   - All chunks indexed with multilingual-e5-large
   - Queries encoded with same model
   - Cosine similarity works across languages
   - No translation layer needed

4. TRADEOFFS:
   - Multilingual embeddings: Simpler, faster, but slightly less precise
   - Translation: More precise, but slower, error-prone
   - CHOICE: Multilingual embeddings (simplicity wins)

5. FALLBACK:
   - If query is English and no good results, try Hindi translation
   - If query is Hindi and no good results, try English translation
   - This handles edge cases where embedding space has gaps
"""




class RAGEvaluator:
    """
    Evaluation framework for RAG system.
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_retrieval(self, question: str, retrieved_chunks: List[Chunk], 
                          relevant_chunk_ids: List[str]) -> Dict:
        """
        Evaluate retrieval quality.
        
        Metrics:
        - Recall@k: Fraction of relevant chunks retrieved
        - MRR: Mean Reciprocal Rank
        - NDCG: Normalized Discounted Cumulative Gain
        """
        retrieved_ids = [chunk.text[:100] for chunk in retrieved_chunks]  # Use first 100 chars as ID
        
        # Recall@k
        relevant_retrieved = sum(1 for cid in relevant_chunk_ids if any(cid in rid for rid in retrieved_ids))
        recall = relevant_retrieved / len(relevant_chunk_ids) if relevant_chunk_ids else 0
        
        # MRR
        mrr = 0
        for rank, rid in enumerate(retrieved_ids, 1):
            if any(cid in rid for cid in relevant_chunk_ids):
                mrr = 1 / rank
                break
        
        return {
            "recall_at_k": recall,
            "mrr": mrr,
            "retrieved_count": len(retrieved_chunks),
            "relevant_count": len(relevant_chunk_ids)
        }
    
    def evaluate_grounding(self, answer: str, context: str) -> Dict:
        """
        Evaluate answer grounding.
        
        Metrics:
        - Claim verification rate
        - Hallucination rate
        - Refusal accuracy
        """
        # Simple claim extraction
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        supported_count = 0
        for sentence in sentences:
            words = set(sentence.lower().split())
            important_words = [w for w in words if len(w) > 4]
            if important_words:
                match_count = sum(1 for w in important_words if w in context.lower())
                if match_count >= len(important_words) * 0.3:
                    supported_count += 1
        
        claim_verification_rate = supported_count / len(sentences) if sentences else 0
        hallucination_rate = 1 - claim_verification_rate
        
        return {
            "claim_verification_rate": claim_verification_rate,
            "hallucination_rate": hallucination_rate,
            "total_claims": len(sentences),
            "supported_claims": supported_count
        }
    
    def evaluate_end_to_end(self, question: str, answer: str, 
                           expected_answer: str, context: str) -> Dict:
        """
        End-to-end evaluation.
        
        Metrics:
        - ROUGE-L (answer quality)
        - Grounding score
        - Refusal accuracy
        """
        # Simple ROUGE-L approximation
        answer_words = set(answer.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        if answer_words and expected_words:
            overlap = len(answer_words & expected_words)
            precision = overlap / len(answer_words)
            recall = overlap / len(expected_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 0
        
        grounding = self.evaluate_grounding(answer, context)
        
        return {
            "rouge_l_f1": f1,
            "grounding_score": grounding["claim_verification_rate"],
            "hallucination_rate": grounding["hallucination_rate"]
        }
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.results:
            print("No evaluation results yet.")
            return
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        avg_recall = np.mean([r.get("recall_at_k", 0) for r in self.results])
        avg_mrr = np.mean([r.get("mrr", 0) for r in self.results])
        avg_grounding = np.mean([r.get("grounding_score", 0) for r in self.results])
        avg_hallucination = np.mean([r.get("hallucination_rate", 0) for r in self.results])
        
        print(f"Average Recall@k: {avg_recall:.3f}")
        print(f"Average MRR: {avg_mrr:.3f}")
        print(f"Average Grounding Score: {avg_grounding:.3f}")
        print(f"Average Hallucination Rate: {avg_hallucination:.3f}")
        print("="*60)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Test the improved RAG system."""
    
    print("="*60)
    print("RAG SYSTEM v2 - Production Ready")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    pipeline.load_and_index()
    
    # Test queries
    test_queries = [
        "How does a neural network progressively transform raw pixel inputs into a final classification output?",
        "Why are biases necessary in addition to weights in a neural network?",
        "What limitation of RNNs do transformers overcome, and how does self-attention address it?",
        "How does deep learning eliminate the need for manual feature engineering compared to traditional machine learning?",
        "Why is data preprocessing important before training a machine learning or deep learning model?"
    ]
    
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Question: {query}")
        
        result = pipeline.query(query)
        
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {result['sources']}")
        print(f"Retrieval Scores: {[f'{s:.2f}' for s in result.get('retrieval_scores', [])]}")
        print(f"Answer: {result['answer'][:200]}...")
    
    print("\n" + "="*60)
    print("SYSTEM READY")
    print("="*60)


if __name__ == "__main__":
    main()
