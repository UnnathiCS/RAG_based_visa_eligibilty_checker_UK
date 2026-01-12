"""
RAG System for UK Visa Policies using FAISS + Mistral
Separate implementation for UK policies with independent chunking & storage.

Quick Setup:
1. pip install faiss-cpu sentence-transformers PyPDF2 requests torch numpy
2. ollama serve (in another terminal)
3. python rag_system_uk.py
"""

import os
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import requests

try:
    import faiss
except:
    print("Installing FAISS...")
    os.system("pip install faiss-cpu")
    import faiss


class UKVisaRAGSystem:
    """RAG system specifically for UK Visa Policies"""
    
    def __init__(self, pdf_paths, db_path="./uk_visa_db"):
        self.pdf_paths = pdf_paths if isinstance(pdf_paths, list) else [pdf_paths]
        self.db_path = db_path
        self.chunks_file = f"{db_path}/chunks.pkl"
        self.index_file = f"{db_path}/faiss.index"
        self.metadata_file = f"{db_path}/metadata.json"
        
        # Create db directory
        os.makedirs(db_path, exist_ok=True)
        
        # Load embeddings model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded\n")
        
        self.index = None
        self.chunks = []
        self.metadata = []
        self.ollama_url = "http://localhost:11434/api/generate"

    def load_from_disk(self):
        """Load previously stored chunks and index from disk"""
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            print(" Loading existing UK visa database...")
            self.index = faiss.read_index(self.index_file)
            self.chunks = pickle.load(open(self.chunks_file, 'rb'))
            self.metadata = json.load(open(self.metadata_file, 'r'))
            print(f"✓ Loaded {len(self.chunks)} chunks from disk\n")
            return True
        return False

    def extract_pdf(self, pdf_path):
        """Extract text from a single UK visa policy PDF"""
        if not os.path.exists(pdf_path):
            print(f"⚠️  File not found: {pdf_path}")
            return []
        
        print(f" Extracting: {Path(pdf_path).name}")
        pages_text = []
        
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append({
                        'page': page_num + 1,
                        'text': text,
                        'source': Path(pdf_path).name
                    })
            
            print(f"   ✓ Extracted {len(pages_text)} pages\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
        
        return pages_text

    def extract_all_pdfs(self):
        """Extract text from all UK policy PDFs"""
        print("="*60)
        print("Extracting UK Visa Policy PDFs")
        print("="*60 + "\n")
        
        all_pages = []
        for pdf_path in self.pdf_paths:
            pages = self.extract_pdf(pdf_path)
            all_pages.extend(pages)
        
        print(f"✓ Total pages extracted: {len(all_pages)}\n")
        return all_pages

    def chunk_text(self, pages, chunk_size=350, overlap=50):
        """
        Chunk UK visa policy text with sliding window approach
        
        Args:
            pages: List of page dicts with 'page', 'text', 'source' keys
            chunk_size: Characters per chunk (~350)
            overlap: Character overlap between chunks (~50)
        
        Returns:
            List of chunk dicts with content and metadata
        """
        print("="*60)
        print(f"Chunking Configuration: size={chunk_size}, overlap={overlap}")
        print("="*60 + "\n")
        
        # UK visa keywords for metadata tagging
        uk_keywords = {
            'visa_type': [
                'Student Visa', 'Skilled Worker', 'Health and Care Worker',
                'Graduate Visa', 'Family Visa', 'Standard Visitor', 'Work Visa',
                'Entrepreneur', 'Innovator', 'Intra-company Transfer', 'Temporary Migration'
            ],
            'section': [
                'Eligibility', 'Requirements', 'Application', 'Fee', 'Validity', 
                'Extensions', 'Conditions', 'Dependants', 'Income', 'Qualifications',
                'Sponsor', 'Points', 'Assessment', 'Processing'
            ],
            'process': [
                'Apply', 'Prove', 'Decision', 'Processing time', 'Appeal', 
                'Interview', 'Documentary evidence', 'Biometric'
            ]
        }
        
        chunks_list = []
        chunk_id = 0
        
        print("Creating chunks...\n")
        
        # Process each page
        for page_info in pages:
            full_text = page_info['text']
            source = page_info.get('source', 'Unknown')
            page_num = page_info.get('page', 0)
            
            # Sliding window chunking
            step = chunk_size - overlap
            for i in range(0, len(full_text), step):
                chunk_text = full_text[i:i + chunk_size]
                
                # Skip very small chunks
                if len(chunk_text.split()) < 15:
                    continue
                
                # Detect metadata from keywords
                visa_type = 'General UK Visa'
                section = 'General'
                process = 'Other'
                
                chunk_lower = chunk_text.lower()
                
                # Match visa type
                for vtype in uk_keywords['visa_type']:
                    if vtype.lower() in chunk_lower:
                        visa_type = vtype
                        break
                
                # Match section
                for sec in uk_keywords['section']:
                    if sec.lower() in chunk_lower:
                        section = sec
                        break
                
                # Match process
                for proc in uk_keywords['process']:
                    if proc.lower() in chunk_lower:
                        process = proc
                        break
                
                metadata = {
                    'id': chunk_id,
                    'visa_type': visa_type,
                    'section': section,
                    'process': process,
                    'source': source,
                    'page': page_num,
                    'char_start': i,
                    'char_end': i + len(chunk_text),
                    'word_count': len(chunk_text.split())
                }
                
                chunks_list.append({
                    'id': chunk_id,
                    'content': chunk_text,
                    'metadata': metadata
                })
                
                chunk_id += 1
        
        print(f"✓ Created {len(chunks_list)} UK visa chunks\n")
        return chunks_list

    def store_chunks(self, chunks_data):
        """Store chunks and embeddings in FAISS"""
        print("="*60)
        print("Storing Chunks and Building FAISS Index")
        print("="*60 + "\n")
        
        # Extract chunk texts
        chunk_texts = [c['content'] for c in chunks_data]
        
        # Generate embeddings in batches
        print(f" Embedding {len(chunk_texts)} chunks...")
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            batch_emb = self.embedder.encode(batch, convert_to_numpy=True).astype('float32')
            embeddings.extend(batch_emb)
            if (i + batch_size) % (batch_size * 5) == 0:
                print(f"   Processed {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)} chunks")
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks = chunk_texts
        self.metadata = [c['metadata'] for c in chunks_data]
        
        # Save to disk
        faiss.write_index(self.index, self.index_file)
        pickle.dump(self.chunks, open(self.chunks_file, 'wb'))
        json.dump(self.metadata, open(self.metadata_file, 'w'), indent=2)
        
        print(f"\n✓ Stored in {self.db_path}/")
        print(f"   - faiss.index ({self.index.ntotal} vectors, {dimension}D)")
        print(f"   - chunks.pkl ({len(self.chunks)} texts)")
        print(f"   - metadata.json ({len(self.metadata)} entries)\n")

    def query(self, question, k=5):
        """Query the UK visa RAG system"""
        if self.index is None or len(self.chunks) == 0:
            print("❌ Database not initialized. Run setup_and_run() first.")
            return
        
        print(f"\n Query: {question}\n")
        
        # Embed question
        question_emb = self.embedder.encode([question]).astype('float32')
        
        # Search FAISS
        distances, indices = self.index.search(question_emb, k)
        
        print(f" Retrieved {k} most relevant chunks:\n")
        
        # Build context from top chunks
        context = ""
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                chunk_text = self.chunks[idx]
                meta = self.metadata[idx]
                context += f"[From {meta.get('source')} - Page {meta.get('page')}]\n{chunk_text}\n\n"
                
                print(f"   [{rank+1}] Dist: {dist:.4f} | Type: {meta.get('visa_type')} | Section: {meta.get('section')} | Source: {meta.get('source')}")
        
        # Query Mistral via Ollama
        print(f"\n Generating answer from Mistral...\n")
        
        prompt = f"""You are an expert on UK visa policies. Answer the following question based on the provided UK visa policy document excerpts.

Question: {question}

UK Visa Policy Context:
{context}

Answer: """
        
        payload = {
            "model": "mistral:latest",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                answer = response.json().get('response', 'No response generated')
                print(f"Answer:\n{answer}\n")
                return answer
            else:
                print(f"❌ Ollama error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Make sure 'ollama serve' is running.")
        except Exception as e:
            print(f"❌ Error: {e}")

    def setup_and_run(self):
        """Full pipeline: extract → chunk → store → query"""
        
        # Check if data already exists
        if self.load_from_disk():
            print("UK visa database already exists!\n")
        else:
            # Extract all PDFs
            pages = self.extract_all_pdfs()
            
            if not pages:
                print("❌ No pages extracted. Check PDF paths.")
                return
            
            # Chunk text
            chunks_data = self.chunk_text(pages)
            
            # Store in FAISS
            self.store_chunks(chunks_data)
        
        # Interactive query loop
        print("="*60)
        print("🇬🇧 UK VISA RAG SYSTEM - Ready for Queries")
        print("="*60)
        print("Type 'exit' to quit\n")
        
        while True:
            user_query = input("Ask about UK visas: ").strip()
            if user_query.lower() == 'exit':
                print("\n Goodbye!")
                break
            if user_query:
                self.query(user_query, k=5)


if __name__ == "__main__":
    # UK visa policy PDFs
    pdf_paths = [
        "/Users/unnathics/Documents/INTERNSHIP/INTERNSHIP/INFOSYS_SPRINGBOARD/dataset_infosys/Print Student visa - GOV.UK.pdf",
        "/Users/unnathics/Documents/INTERNSHIP/INTERNSHIP/INFOSYS_SPRINGBOARD/dataset_infosys/Print Skilled Worker visa - GOV.UK.pdf",
        "/Users/unnathics/Documents/INTERNSHIP/INTERNSHIP/INFOSYS_SPRINGBOARD/dataset_infosys/Print Graduate visa - GOV.UK.pdf",
        "/Users/unnathics/Documents/INTERNSHIP/INTERNSHIP/INFOSYS_SPRINGBOARD/dataset_infosys/Print Health and Care Worker visa - GOV.UK.pdf",
        "/Users/unnathics/Documents/INTERNSHIP/INTERNSHIP/INFOSYS_SPRINGBOARD/dataset_infosys/Print Visit the UK as a Standard Visitor - GOV.UK.pdf",
    ]
    
    # Initialize and run
    rag = UKVisaRAGSystem(pdf_paths=pdf_paths, db_path="./uk_visa_db")
    rag.setup_and_run()
