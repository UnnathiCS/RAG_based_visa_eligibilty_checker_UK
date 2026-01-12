# 🇬🇧 UK Visa RAG System

Separate RAG pipeline for multiple UK visa policies with independent chunking, FAISS vector storage, and Mistral LLM integration.

## 📋 Overview

This system processes **5 UK visa policy PDFs** into a dedicated vector database:
- Student Visa
- Skilled Worker Visa
- Graduate Visa
- Health and Care Worker Visa
- Standard Visitor (Tourism)

Each PDF is independently chunked, embedded, and indexed for semantic search and retrieval-augmented generation.

## 📁 Files & Storage

### Created Files
- **`rag_system_uk.py`** — Complete UK visa RAG system (340 lines)
- **`uk_visa_db/`** — Persistent vector database (created on first run)
  - `faiss.index` — FAISS vector index (float32 embeddings)
  - `chunks.pkl` — Serialized chunk texts
  - `metadata.json` — Chunk metadata (visa type, section, source, page)

### Input Data
Located in `dataset_infosys/`:
```
Print Student visa - GOV.UK.pdf
Print Skilled Worker visa - GOV.UK.pdf
Print Graduate visa - GOV.UK.pdf
Print Health and Care Worker visa - GOV.UK.pdf
Print Visit the UK as a Standard Visitor - GOV.UK.pdf
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies (if not already installed)
pip install faiss-cpu sentence-transformers PyPDF2 requests torch numpy

# Start Ollama in one terminal
ollama serve

# Verify Mistral model is available
ollama list  # Should show mistral:latest
```

### Run the System
```bash
# In another terminal, run the UK RAG system
python rag_system_uk.py
```

### First Run (~2-3 minutes)
1. Downloads embedding model (~90MB)
2. Extracts all 5 UK policy PDFs
3. Creates chunks (sliding window 350 chars, 50 char overlap)
4. Generates embeddings for each chunk
5. Builds FAISS index
6. Saves to `uk_visa_db/`

### Subsequent Runs (immediate)
- Loads pre-computed FAISS index and chunks from disk
- Ready for queries instantly

## ❓ Example Queries

```
Ask about UK visas: What are the eligibility requirements for a Student Visa?
Ask about UK visas: How long does a Skilled Worker visa last?
Ask about UK visas: What are the financial requirements for sponsoring a spouse?
Ask about UK visas: What documents are needed for a Graduate Visa?
Ask about UK visas: Can I visit the UK as a Standard Visitor if I'm from India?
Ask about UK visas: exit
```

##  Chunking Strategy (UK System)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Chunk Size | 350 characters | Medium-sized semantic units |
| Overlap | 50 characters | Preserve context at boundaries |
| Minimum Words | 15 | Skip noise/formatting artifacts |
| Method | Sliding Window | Deterministic, full coverage |

### Detected Keywords for Metadata

**Visa Types:**
- Student Visa, Skilled Worker, Health and Care Worker, Graduate, Family, Standard Visitor, Work, Entrepreneur, Intra-company Transfer

**Sections:**
- Eligibility, Requirements, Application, Fee, Validity, Extensions, Conditions, Dependants, Income, Qualifications, Sponsor, Points, Assessment, Processing

**Processes:**
- Apply, Prove, Decision, Processing time, Appeal, Interview, Documentary evidence, Biometric

## 🔄 Comparison: India vs UK Systems

| Aspect | India (Annex III) | UK Visas |
|--------|-------------------|----------|
| **Database Folder** | `./visa_db/` | `./uk_visa_db/` |
| **Source PDFs** | 1 (72 pages) | 5 (multi-source) |
| **Chunks** | ~582 | ~1,000-1,500 (est.) |
| **Visa Keywords** | e-Visa, Tourist, Business, Transit | Student, Skilled Worker, Graduate, etc. |
| **Metadata Fields** | ID, type, section, position | ID, type, section, process, source, page |
| **Storage Path** | `./visa_db/` | `./uk_visa_db/` |
| **LLM Model** | Mistral | Mistral |
| **Embedding Model** | all-MiniLM-L6-v2 (384-d) | all-MiniLM-L6-v2 (384-d) |

## 💾 Metadata Example

Each chunk is stored with rich metadata for traceability:

```json
{
  "id": 42,
  "visa_type": "Student Visa",
  "section": "Eligibility",
  "process": "Assessment",
  "source": "Print Student visa - GOV.UK.pdf",
  "page": 3,
  "char_start": 1050,
  "char_end": 1400,
  "word_count": 65
}
```

## 🔍 Retrieval Process

1. **Encode Query:** User question → 384-d embedding (same SentenceTransformer model)
2. **FAISS Search:** Query embedding → k=5 nearest neighbors (L2 distance)
3. **Fetch Context:** Retrieve top 5 chunk texts + metadata
4. **Build Prompt:** Assemble system prompt + chunks + user question
5. **Call Ollama:** POST to Mistral (http://localhost:11434/api/generate)
6. **Return Answer:** LLM-generated response grounded in retrieved chunks

## 🛠️ Troubleshooting

### "Cannot connect to Ollama"
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Check if running
curl http://localhost:11434/api/tags
```

### "Model 'mistral' not found"
```bash
ollama pull mistral
```

### Slow first run?
- First run downloads SentenceTransformer embeddings (~90MB) — cached afterward
- Chunking & embedding ~1,000 chunks takes 1-2 minutes

### "FAISS index file corrupted"
```bash
# Delete and recreate
rm -rf uk_visa_db/
python rag_system_uk.py
```

## 📈 Performance Notes

- **Embedding:** ~50 chunks/second (CPU)
- **FAISS Search:** <1ms for k=5 on 1,000+ vectors
- **LLM Generation:** 5-10 seconds (depends on Mistral speed)
- **Memory:** ~100MB (embeddings + index)

## 🔧 Advanced Usage

### Inspect FAISS Index
```python
import faiss

idx = faiss.read_index('uk_visa_db/faiss.index')
print(f"Vectors: {idx.ntotal}")
print(f"Dimensions: {idx.d}")
```

### Export Chunks
```python
import pickle
import json

chunks = pickle.load(open('uk_visa_db/chunks.pkl', 'rb'))
metadata = json.load(open('uk_visa_db/metadata.json', 'r'))

print(f"Total chunks: {len(chunks)}")
print(f"Sources: {set(m['source'] for m in metadata)}")
```

## 📚 References

- **FAISS:** https://github.com/facebookresearch/faiss
- **Sentence Transformers:** https://www.sbert.net/
- **Ollama:** https://ollama.ai/
- **Mistral Model:** https://mistral.ai/

## 🎯 Next Steps

1. ✅ Run UK RAG system and test queries
2. Compare retrieval quality with India system
3. (Optional) Tune chunk size/overlap for better performance
4. (Optional) Switch to token-aware chunking for quality improvement
5. (Optional) Migrate to Chroma/Qdrant for metadata filtering
