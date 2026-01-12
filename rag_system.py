"""
RAG System Implementation for India Visa Dataset
Combines Semantic Chunking + ChromaDB + Mistral LLM

This script:
1. Extracts PDF content
2. Performs semantic chunking
3. Generates embeddings
4. Stores in ChromaDB
5. Enables RAG queries with Mistral
"""

import os
import json
import re
from pathlib import Path
import chromadb
from chromadb.config import Settings
import PyPDF2
from sentence_transformers import SentenceTransformer
import requests


class IndianVisaRAG:
    """Complete RAG system for India Visa policies."""
    
    def __init__(self, pdf_path, chroma_db_path="./chroma_db"):
        """
        Initialize RAG system.
        
        Args:
            pdf_path: Path to the visa policy PDF
            chroma_db_path: Path to store ChromaDB data
        """
        self.pdf_path = pdf_path
        self.chroma_db_path = chroma_db_path
        
        # Initialize ChromaDB
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_db_path,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        
        # Try to get existing collection or create new
        try:
            self.collection = self.client.get_collection("visa_policies")
            print("✓ Loaded existing ChromaDB collection")
        except:
            self.collection = self.client.create_collection("visa_policies")
            print("✓ Created new ChromaDB collection")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def extract_pdf(self):
        """Extract text and structure from PDF."""
        print(f"\n📄 Extracting PDF: {self.pdf_path}")
        
        pages_content = {}
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"Total pages: {num_pages}")
            
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                pages_content[i + 1] = text
        
        return pages_content
    
    def identify_visa_categories(self, text):
        """Extract visa categories and sections from text."""
        categories = {
            "e-Visa": [],
            "Tourist Visa": [],
            "Business Visa": [],
            "Transit Visa": [],
            "Employment Visa": [],
            "Student Visa": [],
            "Medical Visa": [],
            "Other Visas": []
        }
        return categories
    
    def semantic_chunking(self, pages_content, chunk_size=300, overlap=50):
        """
        Perform semantic chunking on the extracted content.
        
        Args:
            pages_content: Dictionary of page_num: text
            chunk_size: Target tokens per chunk (~4 chars = 1 token)
            overlap: Overlap tokens between chunks
            
        Returns:
            List of chunks with metadata
        """
        print(f"\n✂️ Performing semantic chunking...")
        
        chunks = []
        chunk_id = 0
        
        # Combine all pages
        full_text = "\n\n".join(pages_content.values())
        
        # Split by sections (Main headings are usually Roman numerals)
        section_pattern = r'^\s*[IXV]+\.\s+(.+?)$'
        sections = re.split(section_pattern, full_text, flags=re.MULTILINE)
        
        current_section = "General"
        
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk_text = full_text[i:i + chunk_size]
            
            # Skip very small chunks
            if len(chunk_text.split()) < 20:
                continue
            
            # Try to find section
            for section_name in [
                "e-Visa", "Tourist Visa", "Business Visa", "Transit Visa",
                "Employment Visa", "Student Visa", "Medical Visa"
            ]:
                if section_name in chunk_text[:100]:
                    current_section = section_name
                    break
            
            # Extract key subsections
            subsection = "General"
            if "Eligibility" in chunk_text:
                subsection = "Eligibility"
            elif "Conditions" in chunk_text:
                subsection = "Conditions"
            elif "Fee" in chunk_text or "fees" in chunk_text.lower():
                subsection = "Fees"
            elif "Validity" in chunk_text:
                subsection = "Validity"
            elif "Extension" in chunk_text:
                subsection = "Extension"
            
            # Create chunk with metadata
            chunk = {
                "id": f"chunk_{chunk_id}",
                "content": chunk_text.strip(),
                "metadata": {
                    "visa_category": current_section,
                    "section": subsection,
                    "chunk_size": len(chunk_text.split()),
                    "source": "AnnexIII_01022018.pdf"
                }
            }
            
            chunks.append(chunk)
            chunk_id += 1
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def store_in_chromadb(self, chunks):
        """Store chunks and embeddings in ChromaDB."""
        print(f"\n💾 Storing {len(chunks)} chunks in ChromaDB...")
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = self.embedder.encode(chunk["content"]).tolist()
                
                # Add to ChromaDB
                self.collection.add(
                    ids=[chunk["id"]],
                    embeddings=[embedding],
                    documents=[chunk["content"]],
                    metadatas=[chunk["metadata"]]
                )
                
                if (i + 1) % 10 == 0:
                    print(f"  Stored {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                print(f"  Error storing chunk {i}: {e}")
        
        # Persist to disk
        self.client.persist()
        print(f"✓ All chunks stored and persisted to {self.chroma_db_path}")
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question about visas
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results
    
    def generate_rag_response(self, query):
        """
        Generate response using RAG with Mistral.
        
        Args:
            query: User question
            
        Returns:
            Generated response
        """
        print(f"\n🔍 Query: {query}")
        
        # Retrieve relevant chunks
        results = self.retrieve_relevant_chunks(query, top_k=3)
        
        if not results['documents'] or len(results['documents']) == 0:
            return "No relevant information found in the database."
        
        # Build context from retrieved documents
        context = "\n\n".join(results['documents'][0])
        
        # Create prompt with context
        prompt = f"""You are an expert on Indian visa policies. 
Based on the following context from the India Visa Policy document, answer the user's question accurately.

Context:
{context}

Question: {query}

Answer: """
        
        # Query Mistral via Ollama
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "mistral:latest",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "No response generated")
                return answer
            else:
                return f"Error: Mistral API returned status {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Mistral: {str(e)}"
    
    def run_interactive_rag(self):
        """Run interactive RAG query interface."""
        print("\n" + "="*70)
        print("        🇮🇳 India Visa Policy RAG System")
        print("="*70)
        print("\nAsk questions about Indian visas!")
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                query = input("Your Question: ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                response = self.generate_rag_response(query)
                print(f"\n✅ Answer: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def setup_and_run(self):
        """Complete setup and initialization."""
        print("\n🚀 Starting RAG System Setup...\n")
        
        # Extract PDF
        pages_content = self.extract_pdf()
        
        # Perform chunking
        chunks = self.semantic_chunking(pages_content)
        
        # Check if collection is empty
        if self.collection.count() == 0:
            # Store in ChromaDB
            self.store_in_chromadb(chunks)
        else:
            print(f"✓ Using existing {self.collection.count()} stored chunks")
        
        # Start interactive session
        self.run_interactive_rag()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    pdf_path = "/Users/unnathics/Documents/INTERNSHIP/INTERNSHIP/INFOSYS_SPRINGBOARD/dataset_infosys/AnnexIII_01022018.pdf"
    
    # Initialize and run RAG system
    rag_system = IndianVisaRAG(pdf_path=pdf_path)
    rag_system.setup_and_run()
