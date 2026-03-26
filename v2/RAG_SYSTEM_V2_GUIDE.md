# RAG System v2 - Production Ready Implementation Guide

## Overview

This guide documents the transformation of your RAG system from a "keyword + hallucination demo" to a "grounded, retrieval-first, failure-aware RAG system."

---

## PHASE 1: MINIMAL FIXES (80% of problems)

### 1.1 Retrieval Fix: Semantic Embeddings

**Problem:** Keyword matching fails 4/5 times
**Solution:** Replace with semantic embeddings using `multilingual-e5-large`

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load multilingual model (handles English + Hindi)
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Encode all chunks
texts = [f"passage: {chunk.text}" for chunk in chunks]
embeddings = model.encode(texts)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

# Query
query_embedding = model.encode([f"query: {question}"])
faiss.normalize_L2(query_embedding)
scores, indices = index.search(query_embedding, top_k=5)
```

**Why this works:**
- Semantic understanding (not just word overlap)
- Cross-lingual (English queries find Hindi content)
- Fast (FAISS is optimized for similarity search)

### 1.2 Grounding Fix: Strict Prompting

**Problem:** LLM hallucinates freely
**Solution:** Strict grounding prompt + validation

```python
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
```

**Validation logic:**
```python
def is_insufficient(answer: str) -> bool:
    insufficient_phrases = [
        "insufficient to answer",
        "not enough information",
        "cannot answer",
        "does not contain"
    ]
    return any(phrase in answer.lower() for phrase in insufficient_phrases)
```

### 1.3 Multilingual Fix: Multilingual Embeddings

**Problem:** Hindi content inaccessible to English queries
**Solution:** Use multilingual embeddings (NOT translation)

```python
# multilingual-e5-large embeds English and Hindi in same vector space
# No translation needed!
model = SentenceTransformer('intfloat/multilingual-e5-large')

# English query finds Hindi chunks automatically
query = "What is deep learning?"  # English
results = retriever.retrieve(query)  # Can return Hindi chunks!
```

**Why not translation:**
- Translation adds latency
- Translation errors compound
- Multilingual embeddings are simpler and more robust

---

## PHASE 2: ARCHITECTURE UPGRADE

### Pipeline Flow

```
Query
  ↓
Language Detection
  ↓
Semantic Retrieval (FAISS + multilingual-e5-large)
  ↓
Re-ranking (score-based)
  ↓
Grounded Answer Generation (Ollama + strict prompt)
  ↓
Validation (hallucination detection)
  ↓
Response
```

### Key Components

1. **SemanticRetriever:** FAISS index with multilingual embeddings
2. **GroundedAnswerGenerator:** Ollama with strict grounding prompt
3. **MultilingualHandler:** Language detection + optional translation
4. **FailureGuardrails:** Refusal rules + hallucination detection
5. **RAGEvaluator:** Metrics for retrieval and grounding quality

### Installation

```bash
# Install new dependencies
pip install -r requirements_v2.txt

# This will install:
# - sentence-transformers (for embeddings)
# - faiss-cpu (for vector search)
# - numpy (for array operations)
```

### Usage

```python
from rag_system_v2 import RAGPipeline

# Initialize
pipeline = RAGPipeline()
pipeline.load_and_index()

# Query
result = pipeline.query("What is a neural network?")

print(result["answer"])
print(result["status"])  # PASS, REFUSED, or ERROR
print(result["sources"])
```

---

## PHASE 3: FAILURE GUARDRAILS

### Rule 1: Refuse Bad Retrieval

```python
def should_refuse_retrieval(chunks, threshold=0.3):
    if not chunks:
        return True, "No chunks retrieved"
    
    best_score = chunks[0][1]
    if best_score < threshold:
        return True, f"Score {best_score:.2f} below threshold"
    
    return False, "OK"
```

### Rule 2: Refuse Insufficient Context

```python
def should_refuse_answer(answer, context):
    insufficient_phrases = [
        "insufficient to answer",
        "not enough information",
        "cannot answer"
    ]
    
    for phrase in insufficient_phrases:
        if phrase in answer.lower():
            return True, f"Contains refusal phrase: '{phrase}'"
    
    return False, "OK"
```

### Rule 3: Detect Hallucination

```python
def detect_hallucination(answer, context):
    # Check 1: Answer too long
    if len(answer) > len(context) * 0.8:
        return True, "Answer too long"
    
    # Check 2: External knowledge indicators
    external_patterns = [
        r"according to.*research",
        r"studies have shown",
        r"it is well known that"
    ]
    
    for pattern in external_patterns:
        if re.search(pattern, answer.lower()):
            return True, f"External knowledge: {pattern}"
    
    # Check 3: Unsupported claims
    answer_sentences = re.split(r'[.!?]+', answer)
    unsupported = 0
    
    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        important = [w for w in words if len(w) > 4]
        if important:
            matches = sum(1 for w in important if w in context.lower())
            if matches < len(important) * 0.3:
                unsupported += 1
    
    if unsupported > len(answer_sentences) * 0.5:
        return True, f"{unsupported} unsupported sentences"
    
    return False, "OK"
```

### Rule 4: Detect Off-Topic Retrieval

```python
def detect_bad_retrieval(chunks, question):
    question_words = set(question.lower().split())
    stopwords = {'the', 'a', 'an', 'is', 'are', ...}
    keywords = question_words - stopwords
    
    top_chunk_text = chunks[0][0].text.lower()
    matches = sum(1 for kw in keywords if kw in top_chunk_text)
    
    if matches < len(keywords) * 0.2:
        return True, f"Only {matches}/{len(keywords)} keywords matched"
    
    return False, "OK"
```

---

## PHASE 4: MULTILINGUAL STRATEGY

### Approach: Multilingual Embeddings (NOT translation)

**Why:**
1. **Simpler:** No translation layer needed
2. **Faster:** No API calls for translation
3. **More robust:** No translation errors
4. **Cross-lingual:** English queries find Hindi content automatically

**Implementation:**
```python
# All chunks indexed with multilingual-e5-large
model = SentenceTransformer('intfloat/multilingual-e5-large')

# English query
query_en = "What is deep learning?"
embedding_en = model.encode([f"query: {query_en}"])

# Hindi chunk
chunk_hi = "डीप लर्निंग एक मशीन लर्निंग तकनीक है..."
embedding_hi = model.encode([f"passage: {chunk_hi}"])

# Cosine similarity works across languages!
similarity = np.dot(embedding_en, embedding_hi.T)
```

**Tradeoffs:**

| Approach | Pros | Cons |
|----------|------|------|
| Multilingual Embeddings | Simple, fast, robust | Slightly less precise |
| Translation | More precise | Slower, error-prone |
| **CHOICE** | **Multilingual Embeddings** | **Simplicity wins** |

**Fallback strategy:**
```python
# If English query returns low scores, try Hindi translation
if query_lang == "en" and best_score < 0.4:
    translated = translate(query, "hi")
    results_hi = retriever.retrieve(translated)
    # Merge results
```

---

## PHASE 5: EVALUATION STRATEGY

### Metrics to Track

1. **Retrieval Quality:**
   - Recall@k: Fraction of relevant chunks retrieved
   - MRR: Mean Reciprocal Rank
   - NDCG: Normalized Discounted Cumulative Gain

2. **Grounding Quality:**
   - Claim verification rate
   - Hallucination rate
   - Refusal accuracy

3. **Answer Quality:**
   - ROUGE-L (compared to expected answer)
   - BERTScore (semantic similarity)

### Golden Dataset

Create a test set with:
- 20-30 questions
- Expected answers
- Relevant chunk IDs
- Language (en/hi)

Example:
```json
{
  "question": "What is a neural network?",
  "expected_answer": "A neural network is a logical structure inspired by human brain...",
  "relevant_chunks": ["aircAruvnKk_chunk_5", "fHF22Wxuyw4_chunk_12"],
  "language": "en"
}
```

### Evaluation Code

```python
from rag_system_v2 import RAGEvaluator

evaluator = RAGEvaluator()

for test_case in golden_dataset:
    result = pipeline.query(test_case["question"])
    
    # Evaluate retrieval
    retrieval_metrics = evaluator.evaluate_retrieval(
        test_case["question"],
        retrieved_chunks,
        test_case["relevant_chunks"]
    )
    
    # Evaluate grounding
    grounding_metrics = evaluator.evaluate_grounding(
        result["answer"],
        context
    )
    
    # Evaluate end-to-end
    e2e_metrics = evaluator.evaluate_end_to_end(
        test_case["question"],
        result["answer"],
        test_case["expected_answer"],
        context
    )
    
    evaluator.results.append({**retrieval_metrics, **grounding_metrics, **e2e_metrics})

evaluator.print_summary()
```

### What "Good" Looks Like

| Metric | Target | Current |
|--------|--------|---------|
| Recall@5 | >0.8 | ~0.2 |
| MRR | >0.7 | ~0.2 |
| Grounding Score | >0.9 | ~0.0 |
| Hallucination Rate | <0.1 | ~1.0 |
| Refusal Accuracy | >0.95 | N/A |

---

## MIGRATION GUIDE

### Step 1: Install Dependencies

```bash
pip install -r requirements_v2.txt
```

### Step 2: Test New System

```bash
python rag_system_v2.py
```

### Step 3: Update Flask App

Replace the `/retrieve` and `/generate_answer` routes in `app.py`:

```python
from rag_system_v2 import RAGPipeline

# Initialize once at startup
pipeline = RAGPipeline()
pipeline.load_and_index()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    result = pipeline.query(question)
    return jsonify(result)
```

### Step 4: Add Evaluation

```python
# Run evaluation on golden dataset
python evaluate_rag.py
```

---

## TROUBLESHOOTING

### Issue: "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

### Issue: "FAISS not found"

```bash
pip install faiss-cpu
```

### Issue: "Ollama connection error"

Make sure Ollama is running:
```bash
ollama serve
ollama pull llama3
```

### Issue: "Low retrieval scores"

1. Check if chunks are loaded correctly
2. Verify FAISS index is built
3. Try lowering `min_score` threshold
4. Check if question is in correct language

### Issue: "Answers still hallucinating"

1. Make sure you're using the strict prompt
2. Check temperature is low (0.1)
3. Verify validation logic is working
4. Add more refusal phrases if needed

---

## SUMMARY

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| Retrieval | Keyword matching | Semantic embeddings (FAISS) |
| Grounding | Weak prompt | Strict prompt + validation |
| Multilingual | None | Multilingual embeddings |
| Refusal | None | 4 guardrail rules |
| Evaluation | None | Full metrics framework |

### Minimum Viable Changes

If you can only do ONE thing, do this:

**Replace keyword matching with semantic embeddings.**

This alone will fix 80% of retrieval failures.

### Next Steps

1. Install dependencies: `pip install -r requirements_v2.txt`
2. Test: `python rag_system_v2.py`
3. Integrate into Flask app
4. Create golden dataset
5. Run evaluation
6. Iterate on thresholds

---

## FILES CREATED

- `rag_system_v2.py` - Complete RAG system implementation
- `requirements_v2.txt` - Python dependencies
- `RAG_SYSTEM_V2_GUIDE.md` - This guide

## SUPPORT

For issues or questions, refer to:
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://faiss.ai/
- Ollama: https://ollama.ai/
