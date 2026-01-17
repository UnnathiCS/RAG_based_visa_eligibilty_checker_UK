
# =======================
# UI ENHANCEMENTS (NON-LOGIC)
# Added per request – no business logic modified
# =======================

import streamlit as st

# =======================
# END UI ENHANCEMENTS
# =======================




import streamlit as st
import os
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, date, timedelta
import requests
import pandas as pd
# Heavy ML libraries (sentence-transformers, faiss, numpy) are imported lazily
# inside RAGSystemLoader methods to avoid top-level import failures when
# the environment doesn't have those packages installed.
from rules.student_rules import evaluate_student
from rules.graduate_rules import evaluate as evaluate_graduate
from rules.skilled_worker_rules import evaluate as evaluate_skilled_worker
from rules.health_care_rules import evaluate as evaluate_health_care, HEALTHCARE_SOC_CODES
from rules.visitor_rules_clean import evaluate as evaluate_visitor
try:
    from services.sponsors import get_licensed_sponsor_names, is_licensed_employer, get_licensed_student_provider_names, is_licensed_student_provider
except Exception:
    # graceful fallback if service module isn't available at import time
    def get_licensed_sponsor_names(limit=None):
        return []
    def is_licensed_employer(name: str) -> bool:
        return False
    def get_licensed_student_provider_names(limit=None):
        return []
    def is_licensed_student_provider(name: str) -> bool:
        return False
from services.retrieval import retrieve_policy_chunks
from services.llm import llm_explain
from services.tb import fetch_tb_required_countries
from services.financial import check_financial_requirement, calculate_required_funds, validate_financial_evidence

# Set page config
st.set_page_config(
    page_title="SwiftVisa - AI Visa Eligibility",
    page_icon="https://cdn.iconscout.com/icon/free/png-256/passport-460571.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0f172a;
        --secondary: #1e293b;
        --accent: #14b8a6;
        --accent-hover: #0d9488;
        --muted: #64748b;
        --background: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .visa-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .visa-card:hover {
        border-color: #14b8a6;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.15);
    }
    
    .visa-card.selected {
        border-color: #14b8a6;
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.05) 0%, rgba(20, 184, 166, 0.1) 100%);
    }
    
    /* Progress indicator */
    .progress-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .progress-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
    }
    
    .progress-step.active {
        background: #14b8a6;
        color: white;
    }
    
    .progress-step.completed {
        background: #0f172a;
        color: white;
    }
    
    .progress-step.pending {
        background: #e2e8f0;
        color: #64748b;
    }
    
    /* Form section */
    .form-section {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .section-title {
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #14b8a6;
    }
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .result-eligible {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #22c55e;
    }
    
    .result-partial {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
    }
    
    .result-ineligible {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 1px solid #ef4444;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    /* Legacy card support for existing forms */
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.6rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #14b8a6;
    }
    
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)
# Reusable purpose options (used in Eligibility tab and eligibility-final)
PURPOSE_OPTIONS = [
    'Tourism / holiday',
    'Visit family or friends',
    'Volunteer (up to 30 days with a registered charity)',
    'In transit (pass through to another country)',
    'Business (meetings, interviews)',
    'Permitted paid engagement / event',
    'School exchange programme',
    'Short recreational course (up to 30 days)',
    'Study / placement / exam',
    'Academic, senior doctor or dentist',
    'Medical treatment',
    'Other (specify)'
]

# TB Test Required Countries (from UK Gov website)
TB_TEST_REQUIRED_COUNTRIES = {
    'Afghanistan', 'Algeria', 'Angola', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Belarus', 'Benin',
    'Bhutan', 'Bolivia', 'Botswana', 'Brunei', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cape Verde',
    'Central African Republic', 'Chad', 'Cameroon', 'China', 'Congo', 'Côte d\'Ivoire',
    'Democratic Republic of the Congo', 'Djibouti', 'Dominican Republic', 'East Timor', 'Ecuador',
    'Equatorial Guinea', 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Georgia', 'Ghana', 'Guatemala',
    'Guinea', 'Guinea Bissau', 'Guyana', 'Haiti', 'Hong Kong', 'India', 'Indonesia', 'Iraq',
    'Kazakhstan', 'Kenya', 'Kiribati', 'Kyrgyzstan', 'Laos', 'Lesotho', 'Liberia', 'Macau',
    'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Marshall Islands', 'Mauritania',
    'Micronesia, Federated States of', 'Moldova', 'Mongolia', 'Morocco', 'Mozambique', 'Myanmar (Burma)',
    'Namibia', 'Nepal', 'Niger', 'Nigeria', 'North Korea', 'Pakistan', 'Palau', 'Papua New Guinea',
    'Panama', 'Paraguay', 'Peru', 'Philippines', 'Russia', 'Rwanda', 'São Tomé and Principe',
    'Senegal', 'Sierra Leone', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea',
    'South Sudan', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Tajikistan', 'Tanzania', 'Togo',
    'Thailand', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'Uzbekistan', 'Vanuatu', 'Vietnam',
    'Zambia', 'Zimbabwe'
}

# Passport validation helpers (moved to module level for accessibility)
import re

PASSPORT_FORMATS = {
    "United Kingdom": r"^\d{9}$",
    "United States": r"^\d{9}$",
    "Canada": r"^[A-Z]{2}\d{6}$",
    "India": r"^[A-Z][0-9]{7}$",
    "Australia": r"^[A-Z]\d{7}$",
    "New Zealand": r"^[A-Z]{2}\d{6}$",
    "Germany": r"^[CFGHJKLMNPRTVWXYZ0-9]{9}$",
    "France": r"^\d{2}[A-Z]{2}\d{5}$",
    "Italy": r"^[A-Z0-9]{9}$",
    "Spain": r"^[A-Z0-9]{9}$",
    "Netherlands": r"^[A-Z]{2}\d{7}$",
    "Belgium": r"^[A-Z]{2}\d{6}$",
    "Switzerland": r"^[A-Z]\d{8}$",
    "Sweden": r"^\d{8}$",
    "Norway": r"^\d{8}$",
    "Denmark": r"^\d{9}$",
    "Finland": r"^[A-Z]{2}\d{7}$",
    "Austria": r"^[A-Z]\d{7}$",
    "Ireland": r"^[A-Z0-9]{9}$",
    "Portugal": r"^[A-Z]{2}\d{6}$",
    "Greece": r"^[A-Z]{2}\d{7}$",
    "Poland": r"^[A-Z]{2}\d{7}$",
    "Czech Republic": r"^\d{8}$",
    "Slovakia": r"^\d{8}$",
    "Hungary": r"^[A-Z]{2}\d{6}$",
    "Romania": r"^\d{8}$",
    "Bulgaria": r"^\d{9}$",
    "Croatia": r"^\d{9}$",
    "Serbia": r"^\d{9}$",
    "Slovenia": r"^[A-Z]{2}\d{6}$",
    "Lithuania": r"^[A-Z]{2}\d{6}$",
    "Latvia": r"^[A-Z]{2}\d{6}$",
    "Estonia": r"^[A-Z]{2}\d{6}$",
    "Ukraine": r"^[A-Z]{2}\d{6}$",
    "Russia": r"^\d{9}$",
    "Turkey": r"^[A-Z]\d{8}$",
    "China": r"^[A-Z]\d{8}$",
    "Japan": r"^[A-Z]{2}\d{7}$",
    "South Korea": r"^[A-Z]\d{8}$",
    "Singapore": r"^[A-Z]\d{7}$",
    "Malaysia": r"^[A-Z]\d{8}$",
    "Thailand": r"^[A-Z]{2}\d{7}$",
    "Indonesia": r"^[A-Z]\d{7}$",
    "Philippines": r"^[A-Z]\d{7}$",
    "Brazil": r"^[A-Z]{2}\d{6}$",
    "Mexico": r"^[A-Z]\d{8}$",
    "Argentina": r"^[A-Z]{2}\d{6}$",
    "Chile": r"^[A-Z]{2}\d{6}$",
    "South Africa": r"^\d{9}$",
    "Nigeria": r"^[A-Z]\d{8}$",
    "Kenya": r"^[A-Z]\d{8}$",
    "Egypt": r"^[A-Z]\d{8}$",
    "United Arab Emirates": r"^\d{9}$",
    "Saudi Arabia": r"^[A-Z]\d{8}$",
    "Israel": r"^\d{8}$",
    "Pakistan": r"^[A-Z]{2}\d{7}$",
    "Bangladesh": r"^[A-Z]{2}\d{7}$",
    "Sri Lanka": r"^[A-Z]{2}\d{7}$",
    "Nepal": r"^[A-Z]\d{7}$"
}

DEFAULT_JOB_TITLE_TO_SOC = {
    "Health services and public health managers": "1171",
    "Residential care managers": "1232",
    "Biochemists and biomedical scientists": "2113",
    "Physical scientists": "2114",
    "Generalist medical practitioners": "2211",
    "Physiotherapists": "2221",
    "Occupational therapists": "2222",
    "Speech and language therapists": "2223",
    "Psychotherapists": "2224",
    "Midwifery nurses": "2231",
    "Pharmacists": "2251",
    "Social workers": "2461",
    "Laboratory technicians": "3111",
    "Pharmaceutical technicians": "3212"
}

def build_job_title_to_soc(csv_path: str = 'skilled_worker_soc_codes.csv'):
    """Attempt to build a job-title -> SOC mapping from the CSV."""
    try:
        if not os.path.exists(csv_path):
            return DEFAULT_JOB_TITLE_TO_SOC
        df = pd.read_csv(csv_path, dtype=str)
        if 'Job type' not in df.columns or 'SOC code' not in df.columns:
            return DEFAULT_JOB_TITLE_TO_SOC
        df = df[['SOC code', 'Job type']].dropna()
        df['Job type'] = df['Job type'].astype(str).str.strip()
        df['SOC code'] = df['SOC code'].astype(str).str.strip()
        mapping = df.groupby('Job type')['SOC code'].first().to_dict()
        if not mapping:
            return DEFAULT_JOB_TITLE_TO_SOC
        return mapping
    except Exception:
        return DEFAULT_JOB_TITLE_TO_SOC

JOB_TITLE_TO_SOC = build_job_title_to_soc()

def passport_date_check(issue, expiry, min_valid_months=6):
    today = date.today()
    if expiry is None:
        return False, "Expiry date missing"
    try:
        if isinstance(expiry, str):
            expiry_dt = date.fromisoformat(expiry)
        else:
            expiry_dt = expiry
    except Exception:
        return False, "Expiry date invalid"

    if expiry_dt < today:
        return False, "Passport is expired"
    if (expiry_dt - today).days < 183:
        return False, "Passport does not meet minimum validity requirement"
    return True, "Date validity OK"

def validate_passport(number: str, issuing_country: str, issue_date, expiry_date):
    messages = []
    ok = True
    if not number:
        ok = False
        messages.append("Passport number missing")
    else:
        code = (issuing_country or '').strip()
        if code in PASSPORT_FORMATS:
            try:
                if not re.match(PASSPORT_FORMATS[code], number):
                    ok = False
                    messages.append("Passport number format invalid for issuing country")
            except Exception:
                pass

    date_ok, date_msg = passport_date_check(issue_date, expiry_date)
    if not date_ok:
        ok = False
        messages.append(date_msg)
    return ok, messages


def retrieve_with_rag(failed_rules: list, visa_type: str = None, top_k: int = 3):
    """Module-level RAG retrieval helper.

    This prefers an in-memory RAG system (if loaded in session state) and falls
    back to services.retrieval.retrieve_policy_chunks when RAG isn't available.
    Designed to be callable from any tab.
    """
    results = []
    rag = None
    try:
        rags = st.session_state.get('rag_systems') or {}
        # Prefer UK DB if present; otherwise pick the first usable rag system
        if 'uk' in rags:
            rag = rags.get('uk')
        else:
            for v in (rags.values() if isinstance(rags, dict) else []):
                try:
                    if getattr(v, 'index', None) is not None and len(getattr(v, 'chunks', []) or []) > 0:
                        rag = v
                        break
                except Exception:
                    continue
    except Exception:
        rag = None

    try:
        rag_available = rag is not None and getattr(rag, 'index', None) is not None and len(getattr(rag, 'chunks', []) or []) > 0
    except Exception:
        rag_available = False

    if rag_available:
        for rule in failed_rules:
            query_text = f"{visa_type or ''} policy guidance for {rule} eligibility"
            try:
                answer, chunks = rag.query(query_text, top_k=top_k)
            except Exception:
                chunks = []

            if chunks:
                matched = []
                for c in chunks:
                    meta = c.get('metadata', {}) or {}
                    meta_visa = meta.get('visa_type') or meta.get('visa')
                    if meta_visa and visa_type and str(meta_visa).lower() == str(visa_type).lower():
                        matched.append(c)

                to_use = matched if matched else chunks
                for c in to_use:
                    meta = c.get('metadata', {}) or {}
                    doc_name = meta.get('source') or meta.get('doc') or meta.get('title') or 'Policy document'
                    results.append({
                        'rule': rule,
                        'doc': doc_name,
                        'page': meta.get('page', 'N/A'),
                        'section': meta.get('section', ''),
                        'text': c.get('text', '')
                    })
            else:
                from services.retrieval import retrieve_policy_chunks as fallback
                fb = fallback([rule], visa_type=visa_type, top_k=1)
                results.extend(fb)

        return results[:6]
    else:
        from services.retrieval import retrieve_policy_chunks as fallback
        return fallback(failed_rules, visa_type=visa_type, top_k=top_k)



class RAGSystemLoader:
    """Loader for both India and UK RAG systems"""
    
    def __init__(self, system_type="india", db_path=None):
        self.system_type = system_type
        self.db_path = db_path or ("./visa_db" if system_type == "india" else "./uk_visa_db")
        self.index = None
        self.chunks = []
        self.metadata = []
        self.embedder = None
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def load_model(self):
        """Load sentence transformer model"""
        with st.spinner(" Loading embedding model..."):
            try:
                # Import lazily to avoid top-level dependency failures
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                return True
            except Exception as e:
                st.error(f"⚠️ Could not load embedding model: {e}")
                self.embedder = None
                return False
    
    def load_from_disk(self):
        """Load FAISS index and chunks from disk"""
        if not all([
            os.path.exists(f"{self.db_path}/faiss.index"),
            os.path.exists(f"{self.db_path}/chunks.pkl"),
            os.path.exists(f"{self.db_path}/metadata.json")
        ]):
            return False
        
        try:
            # Load FAISS index (import lazily)
            try:
                import faiss
            except Exception as e:
                st.error(f"⚠️ faiss not available: {e}")
                return False

            self.index = faiss.read_index(f"{self.db_path}/faiss.index")
            
            # Load chunks
            with open(f"{self.db_path}/chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load metadata
            with open(f"{self.db_path}/metadata.json", "r") as f:
                self.metadata = json.load(f)
            
            return len(self.chunks) > 0
        except Exception as e:
            st.error(f"❌ Error loading database: {e}")
            return False
    
    def query(self, question, top_k=5):
        """Query the RAG system"""
        if not self.index or len(self.chunks) == 0:
            return None, []
        
        try:
            # Encode question
            if not self.embedder:
                st.error("⚠️ Embedding model not loaded")
                return None, []

            question_embedding = self.embedder.encode([question])[0]
            # Import numpy lazily
            try:
                import numpy as np
            except Exception as e:
                st.error(f"⚠️ numpy not available: {e}")
                return None, []

            question_embedding = np.array([question_embedding]).astype('float32')
            
            # Search FAISS
            distances, indices = self.index.search(question_embedding, top_k)
            
            # Prepare results
            results = []
            for idx, (dist, chunk_idx) in enumerate(zip(distances[0], indices[0])):
                if 0 <= chunk_idx < len(self.chunks):
                    results.append({
                        'rank': idx + 1,
                        'distance': float(dist),
                        'text': self.chunks[chunk_idx],
                        'metadata': self.metadata[chunk_idx] if chunk_idx < len(self.metadata) else {}
                    })
            
            # Generate answer using Mistral
            context = "\n\n".join([f"[Chunk {r['rank']}]\n{r['text']}" for r in results])
            answer = self._generate_answer(question, context)
            
            return answer, results
        except Exception as e:
            st.error(f"❌ Query error: {e}")
            return None, []
    
    def _generate_answer(self, question, context):
        """Generate answer using Mistral via Ollama"""
        try:
            prompt = f"""You are a helpful visa policy assistant. Based on the provided context from visa policies, answer the user's question clearly and accurately.

Context:
{context}

Question: {question}

Answer:"""
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "mistral:latest",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return "⚠️ Could not generate answer from LLM"
        except requests.exceptions.ConnectionError:
            return "⚠️ Ollama not running. Start it with: ollama serve"
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"


def get_system_stats(rag_system):
    """Get statistics about the RAG system"""
    if rag_system.index and len(rag_system.chunks) > 0:
        return {
            'total_chunks': len(rag_system.chunks),
            'vector_dim': rag_system.index.d,
            'db_size': os.path.getsize(f"{rag_system.db_path}/faiss.index") / (1024 * 1024),  # MB
        }
    return None


def display_chunk_result(result, system_type):
    """Display a single chunk result with formatting"""
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"**Rank #{result['rank']}** • Distance: `{result['distance']:.4f}`")
            st.markdown(f"_{result['text'][:300]}..._" if len(result['text']) > 300 else f"_{result['text']}_")
        
        with col2:
            if result['metadata']:
                meta = result['metadata']
                visa_type = meta.get('visa_type', 'N/A')
                section = meta.get('section', 'N/A')
                st.caption(f"**{visa_type}**\n{section}")
                
                if system_type == "uk" and 'source' in meta:
                    st.caption(f" {meta['source']}")
        
        st.divider()


# UI Helper Functions
def render_header():
    st.markdown("""
    <header style="
        position: sticky;
        top: 0;
        z-index: 50;
        width: 100%;
        border-bottom: 1px solid #e2e8f0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        padding: 1rem 0;
        text-align: center;
    ">
        <h1 style="
            font-size: 3rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            line-height: 1.2;
        ">
            SwiftVisa
            <span style="color: #14b8a6;"> AI Eligibility Screening</span>
        </h1>
    </header>
    """, unsafe_allow_html=True)

def render_progress(current_step):
    """Render progress indicator"""
    steps = ['Select Visa', 'Basic Info', 'Visa Details', 'Results']
    cols = st.columns(len(steps))
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.markdown(f"<div style='text-align:center; padding:0.5rem; background:#0f172a; color:white; border-radius:20px;'>✓ {step}</div>", unsafe_allow_html=True)
            elif i == current_step:
                st.markdown(f"<div style='text-align:center; padding:0.5rem; background:#14b8a6; color:white; border-radius:20px;'>{i+1}. {step}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center; padding:0.5rem; background:#e2e8f0; color:#64748b; border-radius:20px;'>{i+1}. {step}</div>", unsafe_allow_html=True)

def calculate_confidence_breakdown(result, visa_type, form_data):
    """Calculate confidence breakdown using 3-layer model"""
    passed_rules = result.get('passed_rules', [])
    failed_rules = result.get('failed_rules', [])
    
    # Convert rules to strings for comparison
    def rule_to_str(rule):
        if isinstance(rule, str):
            return rule.lower()
        elif isinstance(rule, dict):
            return rule.get('rule', str(rule)).lower()
        return str(rule).lower()
    
    passed_str = [rule_to_str(r) for r in passed_rules]
    failed_str = [rule_to_str(r) for r in failed_rules]
    all_rules = passed_rules + failed_rules
    
    # Core Eligibility Rules (70% weight) - keywords to match
    core_keywords = []
    risk_keywords = []
    doc_keywords = []
    
    # Common core keywords
    common_core = ['passport', 'expiry', 'nationality', 'purpose']
    # Common risk keywords
    common_risk = ['refusal', 'criminal', 'history']
    # Common documentation keywords
    common_docs = ['email', 'phone', 'address', 'reference', 'number']
    
    if visa_type == 'Graduate':
        core_keywords = ['currently_in_uk', 'current_uk_visa', 'course_completed', 
                        'education_provider', 'licensed', 'provider_reported', 
                        'student_visa_valid'] + common_core
        risk_keywords = ['expiry', 'close', 'cas'] + common_risk
        doc_keywords = ['cas', 'reference', 'course_level'] + common_docs
    elif visa_type == 'Student':
        core_keywords = ['has_cas', 'cas', 'education_provider', 'licensed', 
                        'course_full_time', 'full_time', 'meets_financial', 
                        'financial_requirement', 'english_requirement'] + common_core
        risk_keywords = ['funds', 'held', '28', 'days', 'course_start'] + common_risk
        doc_keywords = ['cas', 'reference', 'course', 'date', 'duration'] + common_docs
    elif visa_type == 'Skilled Worker':
        core_keywords = ['job_offer', 'employer', 'licensed', 'sponsor', 
                        'certificate_of_sponsorship', 'cos', 'job_is_eligible', 
                        'occupation', 'meets_minimum_salary', 'salary_threshold', 
                        'english_requirement'] + common_core
        risk_keywords = ['criminal', 'certificate', 'required', 'salary', 'barely'] + common_risk
        doc_keywords = ['cos', 'reference', 'soc', 'code', 'job_title'] + common_docs
    elif visa_type == 'Health and Care Worker':
        core_keywords = ['job_offer', 'employer', 'licensed', 'healthcare', 'sponsor',
                        'certificate_of_sponsorship', 'cos', 'job_is_eligible', 
                        'healthcare_role', 'meets_healthcare_salary', 'salary_rules',
                        'english_requirement'] + common_core
        risk_keywords = ['professional', 'registration', 'required'] + common_risk
        doc_keywords = ['registration', 'cos', 'reference'] + common_docs
    elif visa_type == 'Standard Visitor':
        core_keywords = ['purpose', 'permitted', 'visitor', 'stay_within', 
                        '6_months', 'intends_to_leave', 'sufficient_funds'] + common_core
        risk_keywords = ['return', 'travel', 'accommodation', 'arranged'] + common_risk
        doc_keywords = ['purpose', 'travel', 'date'] + common_docs
    
    # Count passed/failed for each category using keyword matching
    def count_category(keywords, passed_list, failed_list):
        category_passed = sum(1 for p in passed_list if any(kw in p for kw in keywords))
        category_failed = sum(1 for f in failed_list if any(kw in f for kw in keywords))
        total = category_passed + category_failed
        # If no rules match, assume all passed (100%)
        if total == 0:
            return 100.0, 0, 0, 0
        return (category_passed / total * 100), category_passed, category_failed, total
    
    core_score, core_passed, core_failed, core_total = count_category(core_keywords, passed_str, failed_str)
    risk_score, risk_passed, risk_failed, risk_total = count_category(risk_keywords, passed_str, failed_str)
    doc_score, doc_passed, doc_failed, doc_total = count_category(doc_keywords, passed_str, failed_str)
    
    # Calculate weighted final score
    final_score = (core_score * 0.7) + (risk_score * 0.2) + (doc_score * 0.1)
    final_score = max(0, min(100, final_score))  # Clamp between 0-100
    
    return {
        'core': {'score': core_score, 'passed': core_passed, 'failed': core_failed, 'total': core_total, 'weight': 0.7},
        'risk': {'score': risk_score, 'passed': risk_passed, 'failed': risk_failed, 'total': risk_total, 'weight': 0.2},
        'docs': {'score': doc_score, 'passed': doc_passed, 'failed': doc_failed, 'total': doc_total, 'weight': 0.1},
        'final': final_score,
        'all_rules': all_rules,
        'passed_rules': passed_rules,
        'failed_rules': failed_rules
    }

def render_visa_selection_cards():
    """Render landing page with hero section and visa type selection - matching the image design"""
    # Hero Section
    st.markdown("""
    <br>
    <div style="text-align: center; margin-bottom: 3rem;">
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 9999px; background: rgba(20, 184, 166, 0.1); color: #14b8a6; font-size: 0.875rem; font-weight: 500; margin-bottom: 1.5rem;">
            ⚡ AI-Powered Immigration Screening
        </div>
        <h1 style="font-size: 3rem; font-weight: 700; color: #0f172a; margin-bottom: 1rem; line-height: 1.2;">
            Check Your UK Visa<br/>
            <span style="color: #14b8a6;">Eligibility Instantly</span>
        </h1>
        <p style="font-size: 1.125rem; color: #64748b; max-width: 42rem; margin: 0 auto 3rem;">
            SwiftVisa uses advanced AI to evaluate your visa eligibility based on official 
            UK immigration policies. Get a comprehensive assessment in minutes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    features = [
        {
            'icon': '🛡️',
            'title': 'Policy-Backed Assessment',
            'description': 'Our AI is grounded in official UK immigration policy documents.'
        },
        {
            'icon': '⚡',
            'title': 'Instant Analysis',
            'description': 'Get your eligibility assessment in seconds, not days.'
        },
        {
            'icon': '🌍',
            'title': 'Multiple Visa Types',
            'description': 'Support for Graduate, Student, Skilled Worker, Health & Care, and Visitor visas.'
        }
    ]
    
    feat_cols = st.columns(3)
    for i, feature in enumerate(features):
        with feat_cols[i]:
            st.markdown(f"""
            <div style="display: flex; align-items: start; gap: 1rem; padding: 1.5rem; border-radius: 12px; background: white; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem;">
                <div style="display: flex; height: 3rem; width: 3rem; align-items: center; justify-content: center; border-radius: 8px; background: rgba(20, 184, 166, 0.1); color: #14b8a6; font-size: 1.5rem; flex-shrink: 0;">
                    {feature['icon']}
                </div>
                <div>
                    <h3 style="font-weight: 600; color: #0f172a; margin-bottom: 0.25rem; font-size: 1rem;">{feature['title']}</h3>
                    <p style="font-size: 0.875rem; color: #64748b; margin: 0;">{feature['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Progress Bar (below features, before visa selection)
    render_progress(0)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Visa Selection Section
    st.markdown("### Select Your Visa Type")
    st.markdown("Choose the visa category that best matches your situation. Our AI will guide you through the specific requirements.")
    
    VISA_TYPES_UI = {
        'Graduate': {
            'icon': '🎓',
            'description': 'For international students who have completed a UK degree',
            'duration': 'Up to 2-3 years'
        },
        'Student': {
            'icon': '📚',
            'description': 'For studying at a UK educational institution',
            'duration': 'Duration of course + extra time'
        },
        'Skilled Worker': {
            'icon': '💼',
            'description': 'For working in the UK with a job offer from a licensed sponsor',
            'duration': 'Up to 5 years'
        },
        'Health and Care Worker': {
            'icon': '🏥',
            'description': 'For healthcare professionals with a job in the NHS or social care',
            'duration': 'Up to 5 years'
        },
        'Standard Visitor': {
            'icon': '✈️',
            'description': 'For tourism, visiting family, or short business trips',
            'duration': 'Up to 6 months'
        }
    }
    
    cols = st.columns(2)
    
    for i, (visa_name, visa_info) in enumerate(VISA_TYPES_UI.items()):
        with cols[i % 2]:
            selected = st.session_state.get('ui_selected_visa') == visa_name
            border_color = "#14b8a6" if selected else "#e2e8f0"
            bg_color = "rgba(20, 184, 166, 0.05)" if selected else "white"
            
            card_html = f"""
            <div style="background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; padding: 1.5rem; margin: 0.5rem 0; transition: all 0.3s ease; cursor: pointer;">
                <h3 style="color: #0f172a; margin-bottom: 0.5rem;">{visa_info['icon']} {visa_name}</h3>
                <p style="color: #64748b; margin-bottom: 0.5rem;">{visa_info['description']}</p>
                <p style="color: #14b8a6; font-size: 0.875rem;">⏱️ {visa_info['duration']}</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            clicked = st.button(f"Select {visa_name}", key=f"visa_btn_{visa_name}", use_container_width=True)
            if clicked:
                st.session_state.ui_selected_visa = visa_name
                st.session_state.ui_current_step = 1
                # Clear any previous form data to start fresh
                if 'ui_form_data' in st.session_state:
                    st.session_state.ui_form_data = {}
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #64748b; font-size: 0.875rem;">
        © 2025 SwiftVisa. This tool provides informational guidance only and does not constitute legal advice.<br/>
        Built with ❤️ by Unnathi CS
    </div>
    """, unsafe_allow_html=True)

def render_basic_common_form(visa_type):
    """Render basic common form for all visa types - EXACT fields from specification"""
    # Initialize form data in session state if not exists
    if 'ui_form_data' not in st.session_state:
        st.session_state.ui_form_data = {}
    
    # Use the same country options as tab-5 for consistency
    country_options = sorted(list(PASSPORT_FORMATS.keys())) + ['Other']
    
    with st.form("basic_common_form"):
        st.markdown('#### Common details')
        # Use Streamlit columns to ensure side-by-side layout inside the form (matching tab-5)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Personal details</div>', unsafe_allow_html=True)
            # Common Entity 1: Full Name (as per passport)
            full_name = st.text_input('Full Name (as per passport) *', value=st.session_state.ui_form_data.get('full_name', ''), key='ui_full_name')
            # Common Entity 2: Date of Birth
            dob = st.date_input('Date of Birth *', 
                               value=st.session_state.ui_form_data.get('dob', date(1990, 1, 1)), 
                               min_value=date(1900,1,1), 
                               max_value=date(2100,12,31),
                               key='ui_dob')
            # Common Entity 3: Nationality
            nationality_saved = st.session_state.ui_form_data.get('nationality', '')
            nationality_idx = 0
            if nationality_saved and nationality_saved in country_options:
                nationality_idx = country_options.index(nationality_saved)
            elif nationality_saved and nationality_saved not in country_options:
                nationality_idx = len(country_options) - 1  # 'Other'
            nationality = st.selectbox('Nationality *', country_options, index=nationality_idx, key='ui_nationality')
            nationality_other = ''
            if nationality == 'Other':
                nationality_other = st.text_input('Please specify nationality', 
                                                  value=st.session_state.ui_form_data.get('nationality_other', ''),
                                                  key='ui_nationality_other')
                if nationality_other:
                    nationality = nationality_other
            # Common Entity 8: Visa Type Applying For (display only, already selected)
            visa_type_applying = st.session_state.ui_selected_visa or 'Not selected'
            st.text_input('Visa Type Applying For', value=visa_type_applying, disabled=True, key='ui_visa_type_display')
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Passport & Travel</div>', unsafe_allow_html=True)
            # Common Entity 4: Passport Number
            passport_number = st.text_input('Passport Number *', value=st.session_state.ui_form_data.get('passport_number', ''), key='ui_passport_number')
            # Common Entity 5: Passport Issue Date
            passport_issue_date = st.date_input('Passport Issue Date *', 
                                               value=st.session_state.ui_form_data.get('passport_issue_date', date.today() - timedelta(days=365)),
                                               min_value=date(1900,1,1), 
                                               max_value=date(2100,12,31),
                                               key='ui_passport_issue')
            # Common Entity 6: Passport Expiry Date
            passport_expiry_date = st.date_input('Passport Expiry Date *', 
                                                value=st.session_state.ui_form_data.get('passport_expiry_date', date.today() + timedelta(days=365*5)),
                                                min_value=date(1900,1,1), 
                                                max_value=date(2100,12,31),
                                                key='ui_passport_expiry')
            # Common Entity 7: Country of Application / Current Location
            currently_in_uk_idx = 1  # Default to 'Outside the UK'
            if st.session_state.ui_form_data.get('currently_in_uk') == 'Inside the UK':
                currently_in_uk_idx = 0
            currently_in_uk = st.selectbox('Country of Application / Current Location', ['Inside the UK', 'Outside the UK'], index=currently_in_uk_idx, key='ui_current_location')
            # Common Entity 8: Purpose of Visit
            purpose_idx = 0
            if st.session_state.ui_form_data.get('purpose_of_visit'):
                saved_purpose = st.session_state.ui_form_data.get('purpose_of_visit')
                if saved_purpose in PURPOSE_OPTIONS:
                    purpose_idx = PURPOSE_OPTIONS.index(saved_purpose)
            purpose_of_visit = st.selectbox('Purpose of Visit', PURPOSE_OPTIONS, index=purpose_idx, key='ui_purpose')
            purpose = purpose_of_visit
            purpose_other = ''
            if purpose_of_visit == 'Other (specify)':
                purpose_other = st.text_input('Please specify purpose', 
                                              value=st.session_state.ui_form_data.get('purpose_other', ''),
                                              key='ui_purpose_other')
                purpose = purpose_other or purpose_of_visit
            # Common Entity 9: Intended Travel / Start Date
            intended_travel_date = st.date_input('Intended Travel / Start Date', 
                                                value=st.session_state.ui_form_data.get('intended_travel_date', date.today() + timedelta(days=30)),
                                                min_value=date.today(), 
                                                max_value=date(date.today().year+2,12,31),
                                                key='ui_intended_travel_date')
            # Common Entity 10: Intended Length of Stay (in months)
            intended_length_of_stay = st.number_input('Intended Length of Stay (months)', 
                                                     min_value=0, 
                                                     max_value=60,
                                                     value=int(st.session_state.ui_form_data.get('intended_length_of_stay', 0)),
                                                     key='ui_intended_length_of_stay')
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Requirements & Contact</div>', unsafe_allow_html=True)
            # Common Entity 11: Funds Available
            funds_available = st.number_input('Funds Available (GBP)', 
                                            min_value=0.0, 
                                            value=float(st.session_state.ui_form_data.get('funds_available', 0.0)),
                                            key='ui_funds')
            
            st.markdown('#### Declarations')
            # Common Entity 12: English Language Requirement Met (Toggle)
            english_toggle = st.session_state.ui_form_data.get('english_requirement') == 'Yes'
            english_requirement_toggle = st.toggle('Have you met the English language requirement (IELTS, TOEFL, etc.)?', 
                                                   value=english_toggle, key='ui_english_toggle')
            english_requirement = 'Yes' if english_requirement_toggle else 'No'
            
            # Common Entity 13: Criminal History Declaration (Toggle)
            criminal_toggle = st.session_state.ui_form_data.get('criminal_history') == 'Yes'
            criminal_history_toggle = st.toggle('Do you have any criminal convictions or pending charges?', 
                                               value=criminal_toggle, key='ui_criminal_toggle')
            criminal_history = 'Yes' if criminal_history_toggle else 'No'
            
            # Common Entity 14: Previous UK Visa Refusal (Toggle)
            refusal_toggle = st.session_state.ui_form_data.get('previous_refusal') == 'Yes'
            previous_refusal_toggle = st.toggle('Have you ever been refused a UK visa or entry to the UK?', 
                                               value=refusal_toggle, key='ui_refusals_toggle')
            previous_refusal = 'Yes' if previous_refusal_toggle else 'No'
            
            st.markdown('#### Contact')
            # Common Entity 15: Email Address
            email = st.text_input('Email Address *', value=st.session_state.ui_form_data.get('email', ''), key='ui_email')
            # Common Entity 16: Phone Number
            phone = st.text_input('Phone Number', value=st.session_state.ui_form_data.get('phone', ''), key='ui_phone')
            # Common Entity 17: Current Address
            current_address = st.text_area('Current Address', value=st.session_state.ui_form_data.get('current_address', ''), key='ui_address')
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns([1, 5])
        with col1:
            back = st.form_submit_button("← Back")
        with col2:
            submit = st.form_submit_button("Continue →", type="primary")
        
        if back:
            st.session_state.ui_current_step = 0
            st.rerun()
        
        if submit:
            # Handle nationality
            final_nationality = nationality
            if nationality == 'Other' and nationality_other:
                final_nationality = nationality_other
            
            # Handle purpose
            final_purpose = purpose
            
            # Save form data - ONLY the exact common entities
            st.session_state.ui_form_data.update({
                'full_name': full_name,
                'dob': dob,
                'nationality': final_nationality,
                'passport_number': passport_number,
                'passport_issue_date': passport_issue_date,
                'passport_expiry_date': passport_expiry_date,
                'country_of_application': currently_in_uk,  # Common Entity 7
                'visa_type_applying_for': visa_type_applying,  # Common Entity 8
                'purpose_of_visit': final_purpose,
                'intended_travel_date': intended_travel_date,
                'intended_length_of_stay': intended_length_of_stay,
                'funds_available': funds_available,
                'english_requirement': english_requirement,
                'criminal_history': criminal_history,
                'previous_refusal': previous_refusal,
                'email': email,
                'phone': phone,
                'current_address': current_address
            })
            
            # Validate required fields
            if full_name and email and final_nationality and passport_number:
                st.session_state.ui_current_step = 2
                st.rerun()
            else:
                st.error("Please fill in all required fields (marked with *)")

def render_visa_specific_form(visa_type):
    """Render visa-specific form based on selected visa type"""
    # Set the visa type in session state for the existing eligibility system
    st.session_state.elig_visa_type = visa_type
    
    # Initialize eligibility form with basic data
    if 'elig_form' not in st.session_state:
        st.session_state.elig_form = {}
    
    # Merge UI form data into eligibility form
    ui_data = st.session_state.get('ui_form_data', {})
    st.session_state.elig_form.update({
        'date_of_birth': ui_data.get('dob'),
        'nationality': ui_data.get('nationality'),
        'currently_in_uk': ui_data.get('currently_in_uk') == 'Inside the UK',
        'passport_issuing_country': ui_data.get('passport_issuing_country'),
        'passport_issue_date': ui_data.get('passport_issue_date'),
        'passport_expiry_date': ui_data.get('passport_expiry_date'),
        'application_date': date.today()
    })
    
    # Initialize eligibility step
    if 'elig_step' not in st.session_state:
        st.session_state.elig_step = 'basic'
    if 'elig_result' not in st.session_state:
        st.session_state.elig_result = None
    if 'elig_retrieved' not in st.session_state:
        st.session_state.elig_retrieved = []
    if 'elig_explanation' not in st.session_state:
        st.session_state.elig_explanation = None
    
    # Render visa-specific form based on type
    if visa_type == 'Student':
        render_student_visa_specific_form()
    elif visa_type == 'Graduate':
        render_graduate_visa_specific_form()
    elif visa_type == 'Skilled Worker':
        render_skilled_worker_visa_specific_form()
    elif visa_type == 'Health and Care Worker':
        render_health_care_visa_specific_form()
    elif visa_type == 'Standard Visitor':
        render_visitor_visa_specific_form()

def render_student_visa_specific_form():
    """Render Student visa specific form - EXACT fields from specification with toggle buttons"""
    with st.form("student_visa_specific_form"):
        st.markdown("#### 📚 Student Visa Requirements")
        
        col1, col2 = st.columns(2)
        with col1:
            # Student Entity 1: has_cas (Toggle)
            has_cas_toggle = st.toggle('Do you have a CAS?', value=False, key='ui_student_has_cas')
            has_cas = 'Yes' if has_cas_toggle else 'No'
            # Student Entity 2: cas_reference_number (ALWAYS SHOW - not conditional)
            cas_reference_number = st.text_input('CAS Reference Number', value=st.session_state.ui_form_data.get('cas_reference_number', ''), key='ui_student_cas_ref')
            # Student Entity 3: education_provider_is_licensed (Yes/No) - EXTRA FIELD as requested
            student_providers = get_licensed_student_provider_names(limit=500)
            if student_providers:
                provider_selected = st.selectbox('Education Provider (Licensed) *', ['-- select --'] + student_providers, key='ui_student_provider')
                if provider_selected == '-- select --':
                    provider_selected = ''
                    education_provider_is_licensed = 'No'
                else:
                    education_provider_is_licensed = 'Yes'
            else:
                provider_selected = ''
                education_provider_is_licensed_toggle = st.toggle('Is Education Provider Licensed?', value=False, key='ui_student_provider_licensed')
                education_provider_is_licensed = 'Yes' if education_provider_is_licensed_toggle else 'No'
            # Student Entity 4: course_level
            course_level = st.selectbox('Course Level', ["Bachelor's", "Master's", 'PhD', 'Foundation', 'Language'], key='ui_student_course_level')
            # Student Entity 5: course_full_time (Toggle)
            course_full_time_toggle = st.toggle('Is the course full-time?', value=False, key='ui_student_course_full_time')
            course_full_time = 'Yes' if course_full_time_toggle else 'No'
        with col2:
            # Student Entity 6: course_start_date
            course_start_date = st.date_input('Course Start Date', key='ui_student_course_start', min_value=date(1900,1,1), max_value=date(2100,12,31))
            # Student Entity 7: course_end_date
            course_end_date = st.date_input('Course End Date', key='ui_student_course_end', min_value=date(1900,1,1), max_value=date(2100,12,31))
            # Student Entity 8: course_duration_months
            course_duration_months = st.number_input('Course Duration (months)', min_value=0, step=1, value=0, key='ui_student_course_duration')
            # Student Entity 9: meets_financial_requirement (Toggle)
            meets_financial_toggle = st.toggle('Do you meet financial requirement?', value=False, key='ui_student_meets_financial')
            meets_financial_requirement = 'Yes' if meets_financial_toggle else 'No'
            # Student Entity 10: funds_held_for_28_days (Toggle)
            funds_held_toggle = st.toggle('Have funds been held for 28 days?', value=False, key='ui_student_funds_28')
            funds_held_for_28_days = 'Yes' if funds_held_toggle else 'No'
            # Student Entity 11: english_requirement_met (Toggle)
            english_met_toggle = st.toggle('Is English requirement met?', value=False, key='ui_student_english_met')
            english_requirement_met = 'Yes' if english_met_toggle else 'No'
        
        st.markdown("---")
        col1, col2 = st.columns([1, 5])
        with col1:
            back = st.form_submit_button("← Back")
        with col2:
            submit = st.form_submit_button("Check Eligibility", type="primary")
        
        if back:
            st.session_state.ui_current_step = 1
            st.rerun()
        
        if submit:
            # Merge common data with visa-specific data (like tab-5 - nested structure)
            common_data = st.session_state.get('ui_form_data', {})
            
            # Prepare data structure matching tab-5 format exactly
            data = {
                "common": {
                    "english_requirement_met": common_data.get("english_requirement") == "Yes",
                    "criminal_history": common_data.get("criminal_history") == "Yes",
                    "previous_refusal": common_data.get("previous_refusal") == "Yes",
                    "funds_available": common_data.get("funds_available", 0)
                },
                "student": {
                    "has_cas": (has_cas == 'Yes') or bool(cas_reference_number),
                    "cas_reference_number": cas_reference_number if cas_reference_number else '',
                    "education_provider_is_licensed": (education_provider_is_licensed == 'Yes') or bool(provider_selected),
                    "course_level": course_level if course_level else "Bachelor's",
                    "course_full_time": (course_full_time == 'Yes'),
                    "course_start_date": course_start_date if course_start_date else date.today() + timedelta(days=30),
                    "course_end_date": course_end_date if course_end_date else date.today() + timedelta(days=365),
                    "course_duration_months": course_duration_months if course_duration_months else 12,
                    "meets_financial_requirement": (meets_financial_requirement == 'Yes'),
                    "funds_held_for_28_days": (funds_held_for_28_days == 'Yes'),
                    "english_requirement_met": common_data.get("english_requirement") == "Yes"
                }
            }
            
            # Also update flat structure for compatibility
            st.session_state.elig_form.update({
                'provider_name': provider_selected if provider_selected else '',
                'provider_is_licensed': (education_provider_is_licensed == 'Yes') or bool(provider_selected),
                'cas_number': cas_reference_number if cas_reference_number else '',
                'has_cas': (has_cas == 'Yes') or bool(cas_reference_number),
                'course_level': course_level if course_level else "Bachelor's",
                'course_full_time': (course_full_time == 'Yes'),
                'course_start_date': course_start_date if course_start_date else date.today() + timedelta(days=30),
                'course_end_date': course_end_date if course_end_date else date.today() + timedelta(days=365),
                'course_duration_months': course_duration_months if course_duration_months else 12,
                'meets_financial_requirement': (meets_financial_requirement == 'Yes'),
                'funds_held_for_28_days': (funds_held_for_28_days == 'Yes'),
                'english_requirement_met': common_data.get("english_requirement") == "Yes"
            })
            
            # Run evaluation using nested structure (like tab-5)
            result = evaluate_student('core', data)
            st.session_state.elig_result = result
            st.session_state.quick_result_shown = True
            
            # Move to results page immediately (LLM will be generated in background on results page)
            st.session_state.ui_current_step = 3
            st.rerun()

def render_graduate_visa_specific_form():
    """Render Graduate visa specific form - EXACT fields from specification with toggle buttons"""
    with st.form("graduate_visa_specific_form"):
        st.markdown("#### 🎓 Graduate Visa Requirements")
        
        col1, col2 = st.columns(2)
        with col1:
            # Graduate Entity 1: currently_in_uk (Toggle)
            currently_in_uk_toggle = st.toggle('Are you currently in the UK?', value=False, key='ui_grad_currently_in_uk')
            currently_in_uk = 'Yes' if currently_in_uk_toggle else 'No'
            # Graduate Entity 2: current_uk_visa_type (Student / Tier 4)
            current_uk_visa_type = st.selectbox('Current UK Visa Type', ['Student', 'Tier 4'], index=0, key='ui_grad_current_visa')
            # Graduate Entity 3: course_completed (Toggle)
            course_completed_toggle = st.toggle('Have you completed your course?', value=False, key='ui_grad_course_completed')
            course_completed = 'Yes' if course_completed_toggle else 'No'
            # Graduate Entity 4: course_level_completed
            course_level_completed = st.selectbox('Course Level Completed', ['RQF3', 'RQF4', 'RQF5', 'RQF6', 'RQF7', 'RQF8'], index=0, key='ui_grad_course_level')
        with col2:
            # Graduate Entity 5: education_provider_is_licensed (Toggle)
            education_provider_is_licensed_toggle = st.toggle('Is Education Provider Licensed?', value=False, key='ui_grad_provider_licensed')
            education_provider_is_licensed = 'Yes' if education_provider_is_licensed_toggle else 'No'
            # Graduate Entity 6: provider_reported_completion_to_home_office (Toggle)
            provider_reported_toggle = st.toggle('Has Provider Reported Completion to Home Office?', value=False, key='ui_grad_provider_reported')
            provider_reported_completion = 'Yes' if provider_reported_toggle else 'No'
            # Graduate Entity 7: original_cas_reference
            original_cas_reference = st.text_input('Original CAS Reference', key='ui_grad_cas_ref')
            # Graduate Entity 8: student_visa_valid_on_application_date (Toggle)
            student_visa_valid_toggle = st.toggle('Is Student Visa Valid on Application Date?', value=False, key='ui_grad_visa_valid')
            student_visa_valid = 'Yes' if student_visa_valid_toggle else 'No'
        
        st.markdown("---")
        col1, col2 = st.columns([1, 5])
        with col1:
            back = st.form_submit_button("← Back")
        with col2:
            submit = st.form_submit_button("Check Eligibility", type="primary")
        
        if back:
            st.session_state.ui_current_step = 1
            st.rerun()
        
        if submit:
            # Merge common data with visa-specific data (like tab-5 - nested structure)
            common_data = st.session_state.get('ui_form_data', {})
            
            # Prepare data structure matching tab-5 format exactly
            data = {
                "common": {
                    "english_requirement_met": common_data.get("english_requirement") == "Yes",
                    "criminal_history": common_data.get("criminal_history") == "Yes",
                    "previous_refusal": common_data.get("previous_refusal") == "Yes",
                    "funds_available": common_data.get("funds_available", 0)
                },
                "graduate": {
                    "currently_in_uk": currently_in_uk == "Yes",
                    "current_uk_visa_type": current_uk_visa_type if current_uk_visa_type else 'Student',
                    "course_completed": course_completed == "Yes",
                    "course_level_completed": course_level_completed if course_level_completed else 'RQF6',
                    "education_provider_is_licensed": education_provider_is_licensed == "Yes",
                    "provider_reported_completion_to_home_office": provider_reported_completion == "Yes",
                    "original_cas_reference": original_cas_reference if original_cas_reference else '',
                    "student_visa_valid_on_application_date": student_visa_valid == "Yes"
                }
            }
            
            # Run evaluation using nested structure (like tab-5)
            result = evaluate_graduate('core', data)
            st.session_state.elig_result = result
            st.session_state.quick_result_shown = True
            
            # Move to results page immediately
            st.session_state.ui_current_step = 3
            st.rerun()

def render_skilled_worker_visa_specific_form():
    """Render Skilled Worker visa specific form - EXACT fields with job title dropdown and SOC mapping"""
    with st.form("skilled_worker_visa_specific_form"):
        st.markdown("#### 💼 Skilled Worker Visa Requirements")
        
        col1, col2 = st.columns(2)
        with col1:
            # SW Entity 1: job_offer_confirmed (Toggle)
            job_offer_confirmed_toggle = st.toggle('Job Offer Confirmed?', value=False, key='ui_sw_job_offer')
            job_offer_confirmed = 'Yes' if job_offer_confirmed_toggle else 'No'
            # SW Entity 2: employer_is_licensed_sponsor (Toggle)
            employer_is_licensed_sponsor_toggle = st.toggle('Is Employer a Licensed Sponsor?', value=False, key='ui_sw_employer_licensed')
            employer_is_licensed_sponsor = 'Yes' if employer_is_licensed_sponsor_toggle else 'No'
            # SW Entity 3: certificate_of_sponsorship_issued (Toggle)
            certificate_of_sponsorship_issued_toggle = st.toggle('Has Certificate of Sponsorship been Issued?', value=False, key='ui_sw_cos_issued')
            certificate_of_sponsorship_issued = 'Yes' if certificate_of_sponsorship_issued_toggle else 'No'
            # SW Entity 4: cos_reference_number
            cos_reference_number = st.text_input('CoS Reference Number', key='ui_sw_cos_ref')
            # SW Entity 5: job_title (Dropdown with SOC mapping - like tab-5)
            job_title_options = sorted(list(JOB_TITLE_TO_SOC.keys()))
            job_title_selected = st.selectbox('Job Title', [''] + job_title_options, key='ui_sw_job_title')
            
            # Auto-update SOC code when job title changes
            if 'sw_soc_code' not in st.session_state:
                st.session_state.sw_soc_code = ''
            if job_title_selected and st.session_state.get('sw_last_job_title') != job_title_selected:
                st.session_state.sw_soc_code = JOB_TITLE_TO_SOC.get(job_title_selected, '')
                st.session_state.sw_last_job_title = job_title_selected
            
            # SW Entity 6: soc_code (auto-filled, editable)
            soc_code = st.text_input('SOC Code (updates automatically according to title)', 
                                    value=st.session_state.sw_soc_code, key='ui_sw_soc_code')
        with col2:
            # SW Entity 7: job_is_eligible_occupation (Toggle)
            job_is_eligible_toggle = st.toggle('Is Job an Eligible Occupation?', value=False, key='ui_sw_job_eligible')
            job_is_eligible_occupation = 'Yes' if job_is_eligible_toggle else 'No'
            # SW Entity 8: salary_offered
            salary_offered = st.number_input('Salary Offered (GBP)', min_value=0, step=100, value=0, key='ui_sw_salary')
            # SW Entity 9: meets_minimum_salary_threshold (Toggle)
            meets_salary_toggle = st.toggle('Meets Minimum Salary Threshold?', value=False, key='ui_sw_meets_salary')
            meets_minimum_salary_threshold = 'Yes' if meets_salary_toggle else 'No'
            # SW Entity 10: english_requirement_met (Toggle)
            english_met_toggle = st.toggle('English Requirement Met?', value=False, key='ui_sw_english_met')
            english_requirement_met = 'Yes' if english_met_toggle else 'No'
            # SW Entity 11: criminal_record_certificate_required (Toggle)
            criminal_required_toggle = st.toggle('Is Criminal Record Certificate Required?', value=False, key='ui_sw_criminal_required')
            criminal_record_certificate_required = 'Yes' if criminal_required_toggle else 'No'
            # SW Entity 12: criminal_record_certificate_provided (Toggle)
            criminal_provided_toggle = st.toggle('Has Criminal Record Certificate been Provided?', value=False, key='ui_sw_criminal_provided')
            criminal_record_certificate_provided = 'Yes' if criminal_provided_toggle else 'No'
        
        st.markdown("---")
        col1, col2 = st.columns([1, 5])
        with col1:
            back = st.form_submit_button("← Back")
        with col2:
            submit = st.form_submit_button("Check Eligibility", type="primary")
        
        if back:
            st.session_state.ui_current_step = 1
            st.rerun()
        
        if submit:
            # Merge common data with visa-specific data (like tab-5 - nested structure)
            common_data = st.session_state.get('ui_form_data', {})
            
            # Prepare data structure matching tab-5 format exactly
            data = {
                "common": {
                    "english_requirement_met": common_data.get("english_requirement") == "Yes",
                    "criminal_history": common_data.get("criminal_history") == "Yes",
                    "previous_refusal": common_data.get("previous_refusal") == "Yes",
                    "funds_available": common_data.get("funds_available", 0)
                },
                "skilled_worker": {
                    "job_offer_confirmed": job_offer_confirmed == "Yes",
                    "employer_is_licensed_sponsor": employer_is_licensed_sponsor == "Yes",
                    "certificate_of_sponsorship_issued": certificate_of_sponsorship_issued == "Yes",
                    "cos_reference_number": cos_reference_number if cos_reference_number else '',
                    "job_title": job_title_selected if job_title_selected else '',
                    "soc_code": soc_code if soc_code else (st.session_state.get('sw_soc_code', '') if job_title_selected else ''),
                    "job_is_eligible_occupation": job_is_eligible_occupation == "Yes",
                    "salary_offered": salary_offered if salary_offered else 0,
                    "meets_minimum_salary_threshold": meets_minimum_salary_threshold == "Yes",
                    "english_requirement_met": common_data.get("english_requirement") == "Yes",
                    "criminal_record_certificate_required": criminal_record_certificate_required == "Yes",
                    "criminal_record_certificate_provided": criminal_record_certificate_provided == "Yes"
                }
            }
            
            # Run evaluation using nested structure (like tab-5)
            result = evaluate_skilled_worker('core', data)
            st.session_state.elig_result = result
            st.session_state.quick_result_shown = True
            
            # Move to results page immediately
            st.session_state.ui_current_step = 3
            st.rerun()

def render_health_care_visa_specific_form():
    """Render Health and Care Worker visa specific form - EXACT fields with job title dropdown and SOC mapping"""
    with st.form("health_care_visa_specific_form"):
        st.markdown("#### 🏥 Health and Care Worker Visa Requirements")
        
        col1, col2 = st.columns(2)
        with col1:
            # HC Entity 1: job_offer_confirmed (Toggle)
            job_offer_confirmed_toggle = st.toggle('Job Offer Confirmed?', value=False, key='ui_hc_job_offer')
            job_offer_confirmed = 'Yes' if job_offer_confirmed_toggle else 'No'
            # HC Entity 2: employer_is_licensed_healthcare_sponsor (Toggle)
            employer_is_licensed_toggle = st.toggle('Is Employer a Licensed Healthcare Sponsor?', value=False, key='ui_hc_employer_licensed')
            employer_is_licensed_healthcare_sponsor = 'Yes' if employer_is_licensed_toggle else 'No'
            # HC Entity 3: certificate_of_sponsorship_issued (Toggle)
            certificate_of_sponsorship_issued_toggle = st.toggle('Has Certificate of Sponsorship been Issued?', value=False, key='ui_hc_cos_issued')
            certificate_of_sponsorship_issued = 'Yes' if certificate_of_sponsorship_issued_toggle else 'No'
            # HC Entity 4: cos_reference_number
            cos_reference_number = st.text_input('CoS Reference Number', key='ui_hc_cos_ref')
            # HC Entity 5: job_title (Dropdown with SOC mapping - like tab-5)
            # Filter to healthcare job titles only
            hc_job_options = sorted([jt for jt, soc in JOB_TITLE_TO_SOC.items() if soc in HEALTHCARE_SOC_CODES])
            if not hc_job_options:
                hc_job_options = sorted(list(DEFAULT_JOB_TITLE_TO_SOC.keys()))
            job_title_selected = st.selectbox('Job Title (choose)', ['-- select --'] + hc_job_options, key='ui_hc_job_title')
            
            # Auto-update SOC code when job title changes
            if 'hc_soc_code' not in st.session_state:
                st.session_state.hc_soc_code = ''
            if job_title_selected and job_title_selected != '-- select --':
                if st.session_state.get('hc_last_job_title') != job_title_selected:
                    st.session_state.hc_soc_code = JOB_TITLE_TO_SOC.get(job_title_selected, DEFAULT_JOB_TITLE_TO_SOC.get(job_title_selected, ''))
                    st.session_state.hc_last_job_title = job_title_selected
            
            # HC Entity 6: soc_code (auto-filled, editable)
            soc_code = st.text_input('SOC Code (auto-filled from job title)', 
                                    value=st.session_state.hc_soc_code, key='ui_hc_soc_code')
        with col2:
            # HC Entity 7: job_is_eligible_healthcare_role (Toggle)
            job_is_eligible_toggle = st.toggle('Is Job an Eligible Healthcare Role?', value=False, key='ui_hc_job_eligible')
            job_is_eligible_healthcare_role = 'Yes' if job_is_eligible_toggle else 'No'
            # HC Entity 8: salary_offered
            salary_offered = st.number_input('Salary Offered (GBP)', min_value=0, step=100, value=0, key='ui_hc_salary')
            # HC Entity 9: meets_healthcare_salary_rules (Toggle)
            meets_salary_toggle = st.toggle('Meets Healthcare Salary Rules?', value=False, key='ui_hc_meets_salary')
            meets_healthcare_salary_rules = 'Yes' if meets_salary_toggle else 'No'
            # HC Entity 10: professional_registration_required (Toggle)
            reg_required_toggle = st.toggle('Is Professional Registration Required?', value=False, key='ui_hc_reg_required')
            professional_registration_required = 'Yes' if reg_required_toggle else 'No'
            # HC Entity 11: professional_registration_provided (Toggle)
            reg_provided_toggle = st.toggle('Has Professional Registration been Provided?', value=False, key='ui_hc_reg_provided')
            professional_registration_provided = 'Yes' if reg_provided_toggle else 'No'
            # HC Entity 12: english_requirement_met (Toggle)
            english_met_toggle = st.toggle('English Requirement Met?', value=False, key='ui_hc_english_met')
            english_requirement_met = 'Yes' if english_met_toggle else 'No'
        
        st.markdown("---")
        col1, col2 = st.columns([1, 5])
        with col1:
            back = st.form_submit_button("← Back")
        with col2:
            submit = st.form_submit_button("Check Eligibility", type="primary")
        
        if back:
            st.session_state.ui_current_step = 1
            st.rerun()
        
        if submit:
            # Merge common data with visa-specific data (like tab-5 - nested structure)
            common_data = st.session_state.get('ui_form_data', {})
            
            # Prepare data structure matching tab-5 format exactly
            final_job_title = job_title_selected if job_title_selected and job_title_selected != '-- select --' else ''
            data = {
                "common": {
                    "english_requirement_met": common_data.get("english_requirement") == "Yes",
                    "criminal_history": common_data.get("criminal_history") == "Yes",
                    "previous_refusal": common_data.get("previous_refusal") == "Yes",
                    "funds_available": common_data.get("funds_available", 0)
                },
                "health_care": {
                    "job_offer_confirmed": job_offer_confirmed == "Yes",
                    "employer_is_licensed_healthcare_sponsor": employer_is_licensed_healthcare_sponsor == "Yes",
                    "certificate_of_sponsorship_issued": certificate_of_sponsorship_issued == "Yes",
                    "cos_reference_number": cos_reference_number if cos_reference_number else '',
                    "job_title": final_job_title,
                    "soc_code": soc_code if soc_code else (st.session_state.get('hc_soc_code', '') if final_job_title else ''),
                    "job_is_eligible_healthcare_role": job_is_eligible_healthcare_role == "Yes",
                    "salary_offered": salary_offered if salary_offered else 0,
                    "meets_healthcare_salary_rules": meets_healthcare_salary_rules == "Yes",
                    "professional_registration_required": professional_registration_required == "Yes",
                    "professional_registration_provided": professional_registration_provided == "Yes",
                    "english_requirement_met": common_data.get("english_requirement") == "Yes"
                }
            }
            
            # Run evaluation using nested structure (like tab-5)
            result = evaluate_health_care('core', data)
            st.session_state.elig_result = result
            st.session_state.quick_result_shown = True
            
            # Move to results page immediately
            st.session_state.ui_current_step = 3
            st.rerun()

def render_visitor_visa_specific_form():
    """Render Standard Visitor visa specific form - EXACT fields with toggle buttons"""
    with st.form("visitor_visa_specific_form"):
        st.markdown("#### ✈️ Standard Visitor Visa Requirements")
        
        col1, col2 = st.columns(2)
        with col1:
            # Visitor Entity 1: purpose_of_visit (already in common form, but needed here for evaluation)
            purpose_of_visit = st.selectbox('Purpose of Visit', PURPOSE_OPTIONS, 
                                           index=0 if not st.session_state.ui_form_data.get('purpose_of_visit') else 
                                           PURPOSE_OPTIONS.index(st.session_state.ui_form_data.get('purpose_of_visit', PURPOSE_OPTIONS[0])),
                                           key='ui_visitor_purpose')
            # Visitor Entity 2: purpose_is_permitted_under_visitor_rules (Toggle)
            purpose_is_permitted_toggle = st.toggle('Is Purpose Permitted Under Visitor Rules?', value=False, key='ui_visitor_purpose_permitted')
            purpose_is_permitted = 'Yes' if purpose_is_permitted_toggle else 'No'
            # Visitor Entity 3: intended_length_of_stay_months
            intended_length_of_stay_months = st.number_input('Intended Length of Stay (months)', min_value=0, max_value=6, value=0, key='ui_visitor_stay_months')
            # Visitor Entity 4: stay_within_6_months_limit (Toggle)
            stay_within_limit_toggle = st.toggle('Is Stay Within 6 Months Limit?', value=False, key='ui_visitor_stay_limit')
            stay_within_6_months_limit = 'Yes' if stay_within_limit_toggle else 'No'
        with col2:
            # Visitor Entity 5: accommodation_arranged (Toggle)
            accommodation_toggle = st.toggle('Is Accommodation Arranged?', value=False, key='ui_visitor_accommodation')
            accommodation_arranged = 'Yes' if accommodation_toggle else 'No'
            # Visitor Entity 6: return_or_onward_travel_planned (Toggle)
            return_travel_toggle = st.toggle('Is Return or Onward Travel Planned?', value=False, key='ui_visitor_return_travel')
            return_or_onward_travel_planned = 'Yes' if return_travel_toggle else 'No'
            # Visitor Entity 7: intends_to_leave_uk_after_visit (Toggle)
            intends_leave_toggle = st.toggle('Do You Intend to Leave UK After Visit?', value=False, key='ui_visitor_intends_leave')
            intends_to_leave_uk_after_visit = 'Yes' if intends_leave_toggle else 'No'
            # Visitor Entity 8: sufficient_funds_for_stay (Toggle)
            sufficient_funds_toggle = st.toggle('Do You Have Sufficient Funds for Stay?', value=False, key='ui_visitor_sufficient_funds')
            sufficient_funds_for_stay = 'Yes' if sufficient_funds_toggle else 'No'
        
        st.markdown("---")
        col1, col2 = st.columns([1, 5])
        with col1:
            back = st.form_submit_button("← Back")
        with col2:
            submit = st.form_submit_button("Check Eligibility", type="primary")
        
        if back:
            st.session_state.ui_current_step = 1
            st.rerun()
        
        if submit:
            # Merge common data with visa-specific data (like tab-5 - nested structure)
            common_data = st.session_state.get('ui_form_data', {})
            
            # Prepare data structure matching tab-5 format exactly
            final_purpose = purpose_of_visit if purpose_of_visit else (common_data.get('purpose_of_visit') or PURPOSE_OPTIONS[0])
            data = {
                "common": {
                    "english_requirement_met": common_data.get("english_requirement") == "Yes",
                    "criminal_history": common_data.get("criminal_history") == "Yes",
                    "previous_refusal": common_data.get("previous_refusal") == "Yes",
                    "funds_available": common_data.get("funds_available", 0)
                },
                "visitor": {
                    "purpose_of_visit": final_purpose,
                    "purpose_is_permitted_under_visitor_rules": purpose_is_permitted == "Yes",
                    "intended_length_of_stay": intended_length_of_stay_months if intended_length_of_stay_months else 0,
                    "stay_within_6_months_limit": stay_within_6_months_limit == "Yes",
                    "accommodation_arranged": accommodation_arranged == "Yes",
                    "return_or_onward_travel_planned": return_or_onward_travel_planned == "Yes",
                    "intends_to_leave_uk_after_visit": intends_to_leave_uk_after_visit == "Yes",
                    "sufficient_funds_for_stay": sufficient_funds_for_stay == "Yes"
                }
            }
            
            # Run evaluation using nested structure (like tab-5)
            result = evaluate_visitor('core', data)
            st.session_state.elig_result = result
            st.session_state.quick_result_shown = True
            
            # Move to results page immediately
            st.session_state.ui_current_step = 3
            st.rerun()

def render_results_display():
    """Render eligibility results with quick result first, then detailed LLM explanation - improved UI"""
    visa_type = st.session_state.ui_selected_visa
    full_name = st.session_state.ui_form_data.get('full_name', 'Applicant')
    nationality = st.session_state.ui_form_data.get('nationality', '')
    
    st.markdown("### Eligibility Assessment")
    st.markdown("Based on the information you provided, here is our AI-powered assessment.")
    
    # Check if we have results from the eligibility system
    if st.session_state.get('elig_result'):
        result = st.session_state.elig_result
        
        # Determine eligibility status
        eligible = result.get('eligible', False) and not result.get('failed_rules')
        has_failed_rules = bool(result.get('failed_rules'))
        
        # Calculate confidence breakdown
        confidence_data = calculate_confidence_breakdown(result, visa_type, st.session_state.ui_form_data)
        final_confidence = int(confidence_data['final'])
        
        # Determine status based on eligibility
        if eligible:
            status = "Eligible"
            status_color = "#22c55e"
            icon = "✅"
            decision_text = "eligible"
        elif has_failed_rules:
            status = "Eligibility Uncertain"
            status_color = "#f59e0b"
            icon = "⚠️"
            decision_text = "not eligible"
        else:
            status = "Partially Eligible"
            status_color = "#f59e0b"
            icon = "⚠️"
            decision_text = "partially eligible"
        
        # Name and Message (BIGGER and ABOVE decision)
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h1 style="font-size: 2.5rem; font-weight: 700; color: #0f172a; margin-bottom: 1rem;">
                Hey <span style="color: #14b8a6;">{full_name}</span>!
            </h1>
            <p style="font-size: 1.5rem; color: #64748b; margin-bottom: 2rem;">
                You are <strong style="color: {status_color};">{decision_text}</strong> for the {visa_type} visa.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Result Box - Status and Confidence Score
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid {status_color}; border-radius: 12px; padding: 2rem; margin: 2rem 0; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
            <h2 style="color: #0f172a; margin-bottom: 1rem; font-size: 1.5rem;">{status}</h2>
            <div style="display: inline-block; padding: 0.5rem 1rem; background: rgba(255,255,255,0.7); border-radius: 9999px; color: {status_color}; font-weight: 600;">
                Confidence Score: <strong>{final_confidence}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence Breakdown Panel (right below the status box)
        with st.expander("🔍 Confidence Breakdown & Decision Provenance", expanded=True):
            core = confidence_data['core']
            risk = confidence_data['risk']
            docs = confidence_data['docs']
            
            st.markdown("#### Category Breakdown")
            
            # Core Eligibility
            core_contribution = core['score'] * core['weight']
            st.markdown(f"""
            **Core Eligibility Rules** (70% weight)
            - Score: {core['score']:.1f}% ({core['passed']}/{core['total']} passed)
            - Weighted Contribution: {core_contribution:.1f}%
            """)
            
            # Risk Factors
            risk_contribution = risk['score'] * risk['weight']
            st.markdown(f"""
            **Risk Factors** (20% weight)
            - Score: {risk['score']:.1f}% ({risk['passed']}/{risk['total']} passed)
            - Weighted Contribution: {risk_contribution:.1f}%
            """)
            
            # Documentation Readiness
            docs_contribution = docs['score'] * docs['weight']
            st.markdown(f"""
            **Documentation Readiness** (10% weight)
            - Score: {docs['score']:.1f}% ({docs['passed']}/{docs['total']} passed)
            - Weighted Contribution: {docs_contribution:.1f}%
            """)
            
            st.markdown(f"""
            **Final Confidence Score: {final_confidence}%**
            """)
            
            # Explainability Table
            st.markdown("#### Why this decision?")
            explain_data = []
            for rule in confidence_data['all_rules']:
                rule_name = rule if isinstance(rule, str) else rule.get('rule', str(rule))
                is_passed = rule in confidence_data['passed_rules']
                category = 'Core' if any(cr in rule_name.lower() for cr in ['passport', 'nationality', 'purpose', 'cas', 'licensed', 'course', 'job', 'salary', 'english']) else \
                          'Risk' if any(rr in rule_name.lower() for rr in ['refusal', 'criminal', 'close', 'barely']) else 'Docs'
                explain_data.append({
                    'Rule': rule_name,
                    'Result': '✅' if is_passed else '❌',
                    'Category': category
                })
            
            if explain_data:
                import pandas as pd
                df = pd.DataFrame(explain_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Generate LLM explanation in background if not already done
        if not st.session_state.get('elig_explanation'):
            if 'llm_generating' not in st.session_state or not st.session_state.llm_generating:
                st.session_state.llm_generating = True
                # Show loading indicator
                with st.spinner('🤖 Generating detailed AI explanation...'):
                    try:
                        retrieved = retrieve_with_rag(result.get('failed_rules', []), visa_type=visa_type)
                        st.session_state.elig_retrieved = retrieved
                        expl = llm_explain({'rule_results': result}, retrieved)
                        st.session_state.elig_explanation = expl
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Could not generate detailed explanation: {e}")
                    finally:
                        st.session_state.llm_generating = False
        
        # Show detailed LLM explanation if available
        if st.session_state.get('elig_explanation'):
            expl = st.session_state.elig_explanation
            
            st.markdown("---")
            st.markdown("#### Detailed AI Analysis")
            
            # Quick reasoning from LLM
            decision = expl.get('decision', '')
            if decision:
                st.markdown(f"""
                <div style="background: #f8fafc; border-left: 4px solid #14b8a6; padding: 1rem; border-radius: 4px; margin-bottom: 1rem;">
                    <strong>Quick Reasoning:</strong> {decision}
                </div>
                """, unsafe_allow_html=True)
            
            # Per-rule explanations (user-friendly)
            if expl.get('per_rule'):
                st.markdown("##### Assessment Details")
                for pr in expl.get('per_rule'):
                    with st.container():
                        st.markdown(f"**{pr.get('rule')}**")
                        st.markdown(f"{pr.get('explanation', '')}")
                        st.markdown("---")
            
            # Summary (if per_rule not available)
            if not expl.get('per_rule') and expl.get('summary'):
                st.markdown("##### Summary")
                st.info(expl.get('summary'))
            
            # Recommendations (with TB test check)
            if expl.get('recommendations'):
                st.markdown("##### Recommendations")
                recommendations = expl.get('recommendations', [])
                
                # Check if TB test is required for this nationality
                if nationality and nationality in TB_TEST_REQUIRED_COUNTRIES:
                    recommendations.append(
                        f"For {nationality}, a TB test is required to enter the UK. "
                        f"Learn more: [UK Government TB Test Requirements](https://www.gov.uk/tb-test-visa/countries-where-you-need-a-tb-test-to-enter-the-uk)"
                    )
                
                for rec in recommendations:
                    st.markdown(f"• {rec}")
        elif has_failed_rules:
            # Show button to generate explanation if not yet generated
            if st.button('Generate Detailed AI Explanation', type="primary"):
                with st.spinner('🤖 Generating detailed AI explanation...'):
                    retrieved = retrieve_with_rag(result.get('failed_rules', []), visa_type=visa_type)
                    st.session_state.elig_retrieved = retrieved
                    expl = llm_explain({'rule_results': result}, retrieved)
                    st.session_state.elig_explanation = expl
                    st.rerun()
    else:
        st.info("Please complete the eligibility check to see results.")
        if st.button("Go to Eligibility Check"):
            st.session_state.ui_current_step = 2
            st.rerun()
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    ⚠️ **Disclaimer**: This assessment is generated by AI for informational purposes only 
    and does not constitute legal advice. Immigration rules are complex and subject to change. 
    Please consult with a qualified immigration lawyer or official government sources before 
    making any decisions about your visa application.
    """)
    
    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Start New Assessment", use_container_width=True):
            st.session_state.ui_current_step = 0
            st.session_state.ui_selected_visa = None
            st.session_state.ui_form_data = {}
            st.session_state.elig_result = None
            st.session_state.elig_explanation = None
            st.session_state.quick_result_shown = False
            st.session_state.llm_generating = False
            st.rerun()
    with col2:
        if st.button("📋 View My Answers", use_container_width=True):
            with st.expander("Your Submitted Information"):
                st.json(st.session_state.ui_form_data)
    with col3:
        st.link_button("🌐 Official UK Visa Info", "https://www.gov.uk/check-uk-visa", use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #64748b; font-size: 0.875rem;">
        Built with ❤️ by Unnathi CS
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize UI state
    if 'ui_current_step' not in st.session_state:
        st.session_state.ui_current_step = 0
    if 'ui_selected_visa' not in st.session_state:
        st.session_state.ui_selected_visa = None
    
    # Render header
    render_header()
    
    # Sidebar Configuration (hidden but functional)
    # Load RAG system in background - don't block UI
    system_type = "UK Visa"
    system_key = "uk"
    
    # Ensure a place to cache loaded RAG systems in session state
    if 'rag_systems' not in st.session_state:
        st.session_state.rag_systems = {}
    
    # Only load RAG if not already loaded and we're past step 0 (when we actually need it)
    if system_key not in st.session_state.rag_systems and st.session_state.ui_current_step > 0:
        with st.sidebar:
            with st.spinner(f"Loading {system_type} RAG system..."):
                rag = RAGSystemLoader(system_type=system_key)
                # Load embedding model (may fail gracefully)
                rag.load_model()
                if rag.load_from_disk():
                    st.session_state.rag_systems[system_key] = rag
                else:
                    st.warning(f"⚠️ Database not found at `{rag.db_path}`. Some features may not work.")
    
    # Get RAG system (may be None if not loaded yet)
    rag = st.session_state.rag_systems.get(system_key)
    
    # Main Content Flow
    if st.session_state.ui_current_step == 0:
        # Step 0: Visa Selection (progress bar is rendered inside render_visa_selection_cards after features)
        render_visa_selection_cards()
        st.stop()
    elif st.session_state.ui_current_step == 1:
        # Step 1: Basic Information Form (common for all visas)
        # Render progress indicator
        render_progress(st.session_state.ui_current_step)
        st.markdown("---")
        
        visa_type = st.session_state.ui_selected_visa
        if not visa_type:
            st.session_state.ui_current_step = 0
            st.rerun()
        
        st.markdown(f"### Step 1: Personal Information - {visa_type} Visa")
        st.markdown("Please provide your personal details as they appear on your passport. All fields marked with * are required.")
        
        # Render basic form (common fields) - matching tab-5 structure
        render_basic_common_form(visa_type)
        st.stop()
    elif st.session_state.ui_current_step == 2:
        # Step 2: Visa-Specific Form
        # Render progress indicator
        render_progress(st.session_state.ui_current_step)
        st.markdown("---")
        
        visa_type = st.session_state.ui_selected_visa
        st.markdown(f"### {visa_type} Requirements")
        st.markdown("Please answer the following questions specific to your visa category.")
        
        # Render visa-specific form
        render_visa_specific_form(visa_type)
        st.stop()
    elif st.session_state.ui_current_step == 3:
        # Step 3: Results
        # Render progress indicator
        render_progress(st.session_state.ui_current_step)
        st.markdown("---")
        
        render_results_display()
        st.stop()
    
    # Legacy tab-based interface (kept for backward compatibility)
    # Main Content Area
    tab4, tab5 = st.tabs([ "Eligibility-Detailed", "Eligibility-Basic"])
   

    # --- Eligibility Check (multi-visa, progressive 3-step flow) ---
    with tab4:
        st.markdown("###  Eligibility Check")

        # Visa type selector (pluggable options)
        visa_type = st.selectbox(
            "Select visa type",
            ["Student", "Graduate", "Skilled Worker", "Health and Care Worker", "Standard Visitor"],
            index=0,
            key='elig_visa_type'
        )

        # Reset step when visa type changes
        if 'elig_step' not in st.session_state:
            st.session_state.elig_step = 'basic'
        if 'elig_form' not in st.session_state:
            st.session_state.elig_form = {}
        if 'elig_result' not in st.session_state:
            st.session_state.elig_result = None
        if 'elig_retrieved' not in st.session_state:
            st.session_state.elig_retrieved = []
        if 'elig_explanation' not in st.session_state:
            st.session_state.elig_explanation = None

        # If visa type changed since last render, reset to basic
        if st.session_state.get('last_visa_type') != visa_type:
            st.session_state.elig_step = 'basic'
            st.session_state.elig_form = {}
            st.session_state.elig_result = None
            st.session_state.elig_retrieved = []
            st.session_state.elig_explanation = None
            st.session_state.last_visa_type = visa_type

        # Simple helper: stub evaluator for non-Student visa types
        def evaluate_stub(step: str, data: dict) -> dict:
            # Minimal stub: pass-through but no failed rules
            return {"eligible": True, "passed_rules": ["STUB_PASS"], "failed_rules": []}

        # Retrieval helper is provided at module level as `retrieve_with_rag`.
        # This avoids duplicate definitions and ensures all tabs can call the
        # same RAG-aware retrieval logic.

        # Short human-readable reasons for rule ids to create a one-line verdict
        RULE_SHORT_REASON = {
            'CAS_PRESENT': 'you do not have a valid CAS (Confirmation of Acceptance for Studies)',
            'CAS_VALID_AGE': 'you must apply for your visa within 6 months of receiving your CAS',
            'PROVIDER_LICENSED': 'the course provider is not a licensed sponsor',
            'COURSE_FULL_TIME': 'the course is not full-time as required',
            'FUNDS_28': 'you have not held the required maintenance funds for 28 consecutive days',
            'ENGLISH_OK': 'you did not meet the required English language standard',
            'ATAS_OK': 'you have not obtained ATAS if required for this course',
            'AGE_OK': 'the applicant does not meet the minimum age requirement'
        }

        # Add financial failure reason mappings
        RULE_SHORT_REASON.update({
            'FUNDS_INSUFFICIENT': 'your available funds are less than the required amount',
            'FUNDS_NOT_HELD_28_DAYS': 'your funds have not been held for the required 28 days',
            'EVIDENCE_TOO_OLD': 'the financial evidence is dated more than 31 days before application',
            'FUNDS_FROM_DISALLOWED_SOURCE': 'the funds are from a disallowed source (crypto, stocks, pensions, overdraft)',
            'FUNDS_HELD_DATE_MISSING': 'the date funds were held is missing',
            'EVIDENCE_DATE_MISSING': 'the financial evidence date is missing',
            'UPLOAD_MISSING': 'financial evidence upload is required but not provided'
        })

        # Navigation helpers
        def step_label(step_key: str) -> str:
            return {
                'basic': 'Basic Check',
                'core': 'Core Check',
                'detailed': 'Detailed Check'
            }.get(step_key, '')

        def step_index(step_key: str) -> int:
            return {'basic': 1, 'core': 2, 'detailed': 3}.get(step_key, 1)

        def can_run_next(result: dict) -> bool:
            return not result.get('failed_rules')

        # Small helper kept as a no-op to avoid large inline UI elements.
        # The user requested removing the large info expander; keeping the
        # function allows future small UX hooks but currently does nothing.
        def field_info_toggle(key: str, message: str):
            return


        # Passport validation helpers for Standard Visitor
        import re

        PASSPORT_FORMATS = {
            "United Kingdom": r"^\d{9}$",
            "United States": r"^\d{9}$",
            "Canada": r"^[A-Z]{2}\d{6}$",
            "India": r"^[A-Z][0-9]{7}$",
            "Australia": r"^[A-Z]\d{7}$",
            "New Zealand": r"^[A-Z]{2}\d{6}$",
            "Germany": r"^[CFGHJKLMNPRTVWXYZ0-9]{9}$",
            "France": r"^\d{2}[A-Z]{2}\d{5}$",
            "Italy": r"^[A-Z0-9]{9}$",
            "Spain": r"^[A-Z0-9]{9}$",
            "Netherlands": r"^[A-Z]{2}\d{7}$",
            "Belgium": r"^[A-Z]{2}\d{6}$",
            "Switzerland": r"^[A-Z]\d{8}$",
            "Sweden": r"^\d{8}$",
            "Norway": r"^\d{8}$",
            "Denmark": r"^\d{9}$",
            "Finland": r"^[A-Z]{2}\d{7}$",
            "Austria": r"^[A-Z]\d{7}$",
            "Ireland": r"^[A-Z0-9]{9}$",
            "Portugal": r"^[A-Z]{2}\d{6}$",
            "Greece": r"^[A-Z]{2}\d{7}$",
            "Poland": r"^[A-Z]{2}\d{7}$",
            "Czech Republic": r"^\d{8}$",
            "Slovakia": r"^\d{8}$",
            "Hungary": r"^[A-Z]{2}\d{6}$",
            "Romania": r"^\d{8}$",
            "Bulgaria": r"^\d{9}$",
            "Croatia": r"^\d{9}$",
            "Serbia": r"^\d{9}$",
            "Slovenia": r"^[A-Z]{2}\d{6}$",
            "Lithuania": r"^[A-Z]{2}\d{6}$",
            "Latvia": r"^[A-Z]{2}\d{6}$",
            "Estonia": r"^[A-Z]{2}\d{6}$",
            "Ukraine": r"^[A-Z]{2}\d{6}$",
            "Russia": r"^\d{9}$",
            "Turkey": r"^[A-Z]\d{8}$",
            "China": r"^[A-Z]\d{8}$",
            "Japan": r"^[A-Z]{2}\d{7}$",
            "South Korea": r"^[A-Z]\d{8}$",
            "Singapore": r"^[A-Z]\d{7}$",
            "Malaysia": r"^[A-Z]\d{8}$",
            "Thailand": r"^[A-Z]{2}\d{7}$",
            "Indonesia": r"^[A-Z]\d{7}$",
            "Philippines": r"^[A-Z]\d{7}$",
            "Brazil": r"^[A-Z]{2}\d{6}$",
            "Mexico": r"^[A-Z]\d{8}$",
            "Argentina": r"^[A-Z]{2}\d{6}$",
            "Chile": r"^[A-Z]{2}\d{6}$",
            "South Africa": r"^\d{9}$",
            "Nigeria": r"^[A-Z]\d{8}$",
            "Kenya": r"^[A-Z]\d{8}$",
            "Egypt": r"^[A-Z]\d{8}$",
            "United Arab Emirates": r"^\d{9}$",
            "Saudi Arabia": r"^[A-Z]\d{8}$",
            "Israel": r"^\d{8}$",
            "Pakistan": r"^[A-Z]{2}\d{7}$",
            "Bangladesh": r"^[A-Z]{2}\d{7}$",
            "Sri Lanka": r"^[A-Z]{2}\d{7}$",
            "Nepal": r"^[A-Z]\d{7}$"
        }

        # Build mapping of job titles -> SOC codes for Healthcare and Skilled Worker dropdowns.
        # Prefer loading the authoritative CSV produced by `skilled_worker_soc_scraper.py`.
        # Fall back to a small curated mapping if the CSV is not present or parsing fails.
        DEFAULT_JOB_TITLE_TO_SOC = {
            "Health services and public health managers": "1171",
            "Residential care managers": "1232",
            "Biochemists and biomedical scientists": "2113",
            "Physical scientists": "2114",
            "Generalist medical practitioners": "2211",
            "Physiotherapists": "2221",
            "Occupational therapists": "2222",
            "Speech and language therapists": "2223",
            "Psychotherapists": "2224",
            "Midwifery nurses": "2231",
            "Pharmacists": "2251",
            "Social workers": "2461",
            "Laboratory technicians": "3111",
            "Pharmaceutical technicians": "3212"
        }

        def build_job_title_to_soc(csv_path: str = 'skilled_worker_soc_codes.csv'):
            """Attempt to build a job-title -> SOC mapping from the CSV.

            The CSV is expected to have columns 'SOC code' and 'Job type'. We pick the
            first SOC code seen for each Job type and return a dict mapping.
            If any error occurs, return the DEFAULT_JOB_TITLE_TO_SOC.
            """
            try:
                if not os.path.exists(csv_path):
                    return DEFAULT_JOB_TITLE_TO_SOC
                df = pd.read_csv(csv_path, dtype=str)
                if 'Job type' not in df.columns or 'SOC code' not in df.columns:
                    return DEFAULT_JOB_TITLE_TO_SOC
                df = df[['SOC code', 'Job type']].dropna()
                # Clean whitespace
                df['Job type'] = df['Job type'].astype(str).str.strip()
                df['SOC code'] = df['SOC code'].astype(str).str.strip()
                # Keep the first SOC code for each job type
                mapping = df.groupby('Job type')['SOC code'].first().to_dict()
                # If mapping empty, fall back to default
                if not mapping:
                    return DEFAULT_JOB_TITLE_TO_SOC
                return mapping
            except Exception:
                return DEFAULT_JOB_TITLE_TO_SOC

        JOB_TITLE_TO_SOC = build_job_title_to_soc()

        def passport_date_check(issue, expiry, min_valid_months=6):
            today = date.today()
            if expiry is None:
                return False, "Expiry date missing"
            try:
                if isinstance(expiry, str):
                    expiry_dt = date.fromisoformat(expiry)
                else:
                    expiry_dt = expiry
            except Exception:
                return False, "Expiry date invalid"

            if expiry_dt < today:
                return False, "Passport is expired"
            # Use 183 days (~6 months) threshold for minimum validity
            if (expiry_dt - today).days < 183:
                return False, "Passport does not meet minimum validity requirement"
            return True, "Date validity OK"

        def validate_passport(number: str, issuing_country: str, issue_date, expiry_date):
            messages = []
            ok = True
            if not number:
                ok = False
                messages.append("Passport number missing")
            else:
                code = (issuing_country or '').strip().upper()
                if code in PASSPORT_FORMATS:
                    try:
                        if not re.match(PASSPORT_FORMATS[code], number):
                            ok = False
                            messages.append("Passport number format invalid for issuing country")
                    except Exception:
                        # if regex fails for any reason, treat as non-fatal
                        pass

            date_ok, date_msg = passport_date_check(issue_date, expiry_date)
            if not date_ok:
                ok = False
                messages.append(date_msg)
            return ok, messages

        # Helper for graduate eligibility: months between two dates (approximate)
        def months_between(start_date, end_date):
            try:
                if not start_date or not end_date:
                    return 0
                # ensure date objects
                if isinstance(start_date, str):
                    start_date = date.fromisoformat(start_date)
                if isinstance(end_date, str):
                    end_date = date.fromisoformat(end_date)
                # approximate months
                return max(0, (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) - (1 if end_date.day < start_date.day else 0))
            except Exception:
                return 0

        def check_graduate_visa(data: dict):
            """Deterministic graduate visa check as described by user.

            Returns a tuple (eligible: bool, passed_rules: list, failed_rules: list)
            where failed_rules contains codes like NOT_IN_UK, NOT_ON_STUDENT_VISA, APPLIED_AFTER_EXPIRY, COURSE_NOT_COMPLETED, COMPLETION_NOT_REPORTED
            """
            failed = []
            passed = []
            # current location
            if data.get('current_location') != 'Inside the UK':
                failed.append('NOT_IN_UK')
            else:
                passed.append('IN_UK')

            if data.get('current_visa_type') != 'Student':
                failed.append('NOT_ON_STUDENT_VISA')
            else:
                passed.append('ON_STUDENT_VISA')

            # dates
            try:
                app_date = data.get('application_date')
                expiry = data.get('student_visa_expiry_date')
                # allow strings
                if isinstance(app_date, str):
                    app_date = date.fromisoformat(app_date)
                if isinstance(expiry, str):
                    expiry = date.fromisoformat(expiry)
                if app_date and expiry and app_date > expiry:
                    failed.append('APPLIED_AFTER_EXPIRY')
                else:
                    passed.append('APPLIED_BEFORE_EXPIRY')
            except Exception:
                # if we can't parse dates, mark as failed conservatively
                failed.append('APPLIED_AFTER_EXPIRY')

            # course duration
            try:
                start = data.get('course_start_date')
                end = data.get('course_end_date')
                if isinstance(start, str):
                    start = date.fromisoformat(start)
                if isinstance(end, str):
                    end = date.fromisoformat(end)
                study_months = months_between(start, end)
                expected = int(data.get('course_expected_duration_months') or 0)
                if study_months < expected:
                    failed.append('COURSE_NOT_COMPLETED')
                else:
                    passed.append('COURSE_COMPLETED')
            except Exception:
                failed.append('COURSE_NOT_COMPLETED')

            if data.get('completion_confirmation') != 'Yes':
                failed.append('COMPLETION_NOT_REPORTED')
            else:
                passed.append('COMPLETION_REPORTED')

            eligible = len(failed) == 0
            return eligible, passed, failed

        # Render progress indicator
        st.markdown(f"**Step {step_index(st.session_state.elig_step)} of 3: {step_label(st.session_state.elig_step)}**")

        # If a non-Student visa is selected, render its own 3-step flow and stop
        if visa_type != 'Student':
            eval_map = {
                'Graduate': evaluate_graduate,
                'Skilled Worker': evaluate_skilled_worker,
                'Health and Care Worker': evaluate_health_care,
                'Standard Visitor': evaluate_visitor
            }
            evaluator = eval_map.get(visa_type)

            # --- BASIC ---
            if st.session_state.elig_step == 'basic':
                with st.form(f'basic_form_{visa_type}'):
                    st.write('#### Basic information')
                    if visa_type == 'Graduate':
                        # Redesigned Graduate Basic inputs
                        current_location = st.selectbox('Where are you applying from?', ['Inside the UK', 'Outside the UK'], key=f'basic_current_location_{visa_type}')
                        current_visa_type = st.selectbox('Current UK visa', ["Student", "Graduate", "Skilled Worker", "Visitor", "Other"], key=f'basic_current_visa_type_{visa_type}')
                        student_visa_expiry_date = st.date_input('Student visa expiry date', key=f'basic_student_visa_expiry_date_{visa_type}', min_value=date(2000,1,1), max_value=date(2100,12,31))
                        # store these into session when form submitted
                    elif visa_type in ('Skilled Worker', 'Health and Care Worker'):
                        has_job_offer = st.checkbox('Do you have a job offer? *', key=f'basic_has_job_offer_{visa_type}')
                        # Provide a dropdown of licensed sponsors where available, fallback to text input
                        # For Skilled Worker include the full list from the CSV (no cap) to allow selecting any employer;
                        # for Health and Care Worker keep a capped list for responsiveness.
                        if visa_type == 'Skilled Worker':
                            sponsor_names = get_licensed_sponsor_names()
                        else:
                            sponsor_names = get_licensed_sponsor_names(limit=500)
                        if sponsor_names:
                            employer_name = st.selectbox('Employer (licensed sponsor) *', ['-- select --'] + sponsor_names, key=f'basic_employer_name_{visa_type}')
                            if employer_name == '-- select --':
                                employer_name = ''
                        else:
                            # If sponsor list unexpectedly empty, show a disabled selectbox placeholder
                            employer_name = st.selectbox('Employer (licensed sponsor) *', ['-- no sponsors available --'], key=f'basic_employer_name_{visa_type}')
                            employer_name = ''

                        # Job title selector populated from the CSV-driven mapping; for Health & Care
                        # only show healthcare-related job types (SOC present in HEALTHCARE_SOC_CODES).
                        if visa_type == 'Health and Care Worker':
                            # Filter JOB_TITLE_TO_SOC to healthcare SOC codes only
                            job_options = sorted([jt for jt, soc in JOB_TITLE_TO_SOC.items() if soc in HEALTHCARE_SOC_CODES])
                            # If the CSV-driven mapping didn't include healthcare titles, fall back to all keys
                            if not job_options:
                                job_options = sorted(list(JOB_TITLE_TO_SOC.keys()))
                        else:
                            job_options = sorted(list(JOB_TITLE_TO_SOC.keys()))
                        job_choice = st.selectbox('Job title / occupation', ['-- select --'] + job_options + ['Other (enter manually)'], key=f'basic_job_title_select_{visa_type}')
                        if job_choice in job_options:
                            job_title = job_choice
                            occupation_code = JOB_TITLE_TO_SOC.get(job_choice, '')
                        elif job_choice == 'Other (enter manually)':
                            job_title = st.text_input('Job title / occupation', key=f'basic_job_title_manual_{visa_type}')
                            occupation_code = st.text_input('Occupation code (SOC) (optional)', key=f'basic_occupation_code_manual_{visa_type}', help='Enter SOC code if known, e.g., 2136')
                        else:
                            job_title = ''
                            occupation_code = ''
                        # For health & care collect COS, english evidence and optional regulator registration at Basic
                        cos_reference = None
                        english_evidence_type = None
                        regulator_registration = None
                        if visa_type == 'Health and Care Worker':
                            cos_reference = st.text_input('Certificate of Sponsorship (CoS) reference', key=f'basic_cos_ref_{visa_type}')
                            english_evidence_type = st.selectbox('English language evidence', [
                                'UK degree taught in English',
                                'Approved SELT test',
                                'Nationality exemption',
                                'Not sure'
                            ], key=f'basic_english_evidence_{visa_type}')
                            regulator_registration = st.checkbox('Regulator registration (e.g., NMC, GMC) if applicable', key=f'basic_regulator_reg_{visa_type}')

                        salary_offered = st.number_input('Salary offered (GBP)', min_value=0, step=100, key=f'basic_salary_offered_{visa_type}')
                    elif visa_type == 'Standard Visitor':
                        # Replace subjective passport question with factual passport fields
                        passport_number = st.text_input('Passport number', key=f'basic_passport_number_{visa_type}')
                        country_options = sorted(list(PASSPORT_FORMATS.keys()))
                        issuing_country = st.selectbox('Issuing country', country_options + ['Other'], key=f'basic_passport_issuing_country_{visa_type}')
                        if issuing_country == 'Other':
                            issuing_country = st.text_input('Please specify issuing country', key=f'basic_passport_issuing_country_other_{visa_type}')

                        passport_issue_date = st.date_input('Passport issue date', key=f'basic_passport_issue_date_{visa_type}', min_value=date(1900,1,1), max_value=date(2100,12,31))
                        passport_expiry_date = st.date_input('Passport expiry date', key=f'basic_passport_expiry_date_{visa_type}', min_value=date(1900,1,1), max_value=date(2100,12,31))
                        # Removed optional passport scan and MRZ fields per updated UX requirements

                        # Collect factual financial amount rather than a boolean
                        funds_available = st.number_input('Amount of funds available for the trip (GBP)', min_value=0, step=50, key=f'basic_funds_available_{visa_type}')
                        # Collect travel dates (factual)
                        planned_departure_date = st.date_input('Planned departure date', key=f'basic_planned_departure_{visa_type}', min_value=date.today(), max_value=date(date.today().year+2,12,31))
                        return_ticket_date = st.date_input('Return ticket date (if purchased)', key=f'basic_return_ticket_{visa_type}', min_value=date.today(), max_value=date(date.today().year+2,12,31))
                        visa_valid_until = st.date_input('Visa valid until (if applicable)', key=f'basic_visa_valid_until_{visa_type}', min_value=date(2000,1,1), max_value=date(2100,12,31))

                    submitted = st.form_submit_button('Run Basic Check', key=f'btn_run_basic_ns_{visa_type}')

                if submitted:
                    st.info(f'Form submitted for {visa_type} (non-student) — running basic checks...')
                    data = {}
                    if visa_type == 'Graduate':
                        data = {
                            'current_location': current_location,
                            'current_visa_type': current_visa_type,
                            'student_visa_expiry_date': student_visa_expiry_date
                        }
                    elif visa_type == 'Skilled Worker':
                        data = {
                            'has_job_offer': has_job_offer,
                            'employer_name': employer_name,
                            'job_title': job_title,
                            'occupation_code': occupation_code,
                            'salary_offered': salary_offered
                        }
                    elif visa_type == 'Health and Care Worker':
                        data = {
                            'has_job_offer': has_job_offer,
                            'employer_name': employer_name,
                            'job_title': job_title,
                            'occupation_code': occupation_code,
                            'salary_offered': salary_offered,
                            'cos_reference': cos_reference,
                            'english_evidence_type': english_evidence_type,
                            'regulator_registration': regulator_registration
                        }
                    elif visa_type == 'Standard Visitor':
                        # Run passport validation and compute booleans expected by rule engine
                        passport_valid, passport_msgs = validate_passport(
                            passport_number,
                            issuing_country,
                            passport_issue_date,
                            passport_expiry_date
                        )

                        # Compute intends_to_leave: prefer explicit return ticket; otherwise rely on visa_valid_until
                        intends_to_leave = bool(return_ticket_date)

                        # Compute sufficient funds: heuristic — require £50 per day if intended stay provided; fallback £500
                        intended_days = st.session_state.elig_form.get('intended_stay_days') or 0
                        try:
                            required = 50 * int(intended_days) if intended_days and int(intended_days) > 0 else 500
                        except Exception:
                            required = 500
                        sufficient_funds = (funds_available >= required)

                        # Compute return travel affordability: true if return ticket present or at least £200 available
                        return_travel_affordable = bool(return_ticket_date) or (funds_available >= 200)

                        data = {
                            'passport_number': passport_number,
                            'passport_issuing_country': issuing_country,
                            'passport_issue_date': passport_issue_date,
                            'passport_expiry_date': passport_expiry_date,
                            
                            'passport_validation': {
                                'valid': passport_valid,
                                'messages': passport_msgs
                            },
                            'valid_passport': passport_valid,
                            'intends_to_leave': intends_to_leave,
                            'funds_available': funds_available,
                            'sufficient_funds': sufficient_funds,
                            'return_ticket_date': return_ticket_date,
                            'return_travel_affordable': return_travel_affordable,
                            'planned_departure_date': planned_departure_date,
                            'visa_valid_until': visa_valid_until
                        }

                    st.session_state.elig_form.update(data)
                    try:
                        result = evaluator('basic', st.session_state.elig_form)
                        # Ensure evaluator returns a dict
                        if not isinstance(result, dict):
                            raise RuntimeError(f"Evaluator returned non-dict: {type(result)}")
                        st.session_state.elig_result = result
                    except Exception as e:
                        # Capture the exception and show it in the UI for debugging
                        st.session_state.elig_result = {"eligible": False, "passed_rules": [], "failed_rules": ["EVALUATION_ERROR"]}
                        st.error('An error occurred while running the basic evaluator. See details below.')
                        st.exception(e)
                        # Reset any retrieved/explanation state to keep UI consistent
                        st.session_state.elig_retrieved = []
                        st.session_state.elig_explanation = None

                # show result
                if st.session_state.get('elig_result'):
                    result = st.session_state.elig_result
                    if result.get('failed_rules'):
                        st.error('❌ Basic checks failed')
                        retrieved = retrieve_with_rag(result.get('failed_rules', []))
                        st.session_state.elig_retrieved = retrieved
                        # Do not call the LLM automatically to avoid blocking the UI.
                        # Show a button that the user can press to request a detailed LLM explanation.
                        if st.session_state.get('elig_explanation'):
                            expl = st.session_state.elig_explanation
                        else:
                            expl = None
                            if st.button('Request detailed LLM explanation', key=f'btn_request_llm_basic_{visa_type}'):
                                with st.spinner('Generating LLM explanation...'):
                                    expl = llm_explain({'rule_results': result}, retrieved)
                                    st.session_state.elig_explanation = expl
                        st.markdown('### LLM Explanation')
                        if expl:
                            st.markdown(f"**Decision:** {expl.get('decision')}")
                            # Prefer per-rule explanations (more detailed) and recommendations
                            if expl.get('per_rule'):
                                st.markdown('#### Per-rule explanations')
                                for pr in expl.get('per_rule'):
                                    st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                    cit = pr.get('citation') or {}
                                    if cit:
                                        st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                        st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                        if cit.get('section'):
                                            st.markdown(f"  - Section: {cit.get('section')}")
                            # Show recommendations prominently
                            if expl.get('recommendations'):
                                st.markdown('#### Recommendations')
                                for rec in expl.get('recommendations'):
                                    st.markdown(f"- {rec}")
                            # Fallback to summary if neither per_rule nor recommendations available
                            if not expl.get('per_rule') and not expl.get('recommendations'):
                                st.markdown(f"**Summary:** {expl.get('summary')}")
                            decision = expl.get('decision', '')
                            if decision and 'not' in decision.lower():
                                failed = result.get('failed_rules', [])
                                reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                                if reasons:
                                    one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                                    st.markdown(f"**Quick reason:** {one_line}")

                        if retrieved:
                            st.markdown('### Supporting citations')
                            for c in retrieved:
                                st.markdown(f"**Document:** {c.get('doc','Unknown')}")
                                st.markdown(f"**Page:** {c.get('page','N/A')}")
                                st.markdown(f"**Section/Paragraph:** {c.get('section','N/A')}")
                    else:
                        st.success('✅ Basic checks passed')
                        st.session_state.elig_retrieved = []
                        st.session_state.elig_explanation = None
                        passed = result.get('passed_rules', [])
                        if passed:
                            st.markdown('**Passed checks:**')
                            for r in passed:
                                st.markdown(f'- ✅ {r}')

                        # For Health & Care Worker, basic is final (no Core/Detailed steps per new requirement)
                        if visa_type != 'Health and Care Worker':
                            if st.button('Proceed to Core check', key=f'btn_proceed_core_{visa_type}'):
                                st.session_state.elig_step = 'core'
                        else:
                            st.info('For Health & Care Worker: the basic check is treated as the final eligibility assessment (no Core/Detailed steps).')

            # --- CORE ---
            elif st.session_state.elig_step == 'core':
                with st.form(f'core_form_{visa_type}'):
                    st.write('#### Core checks')
                    if visa_type == 'Graduate':
                        # Collect factual inputs — provider list sourced from student providers CSV
                        student_providers = get_licensed_student_provider_names(limit=500)
                        if student_providers:
                            provider_name = st.selectbox('Provider name', ['-- select --'] + student_providers, key=f'core_provider_name_{visa_type}', help='Select the sponsoring provider')
                            if provider_name == '-- select --':
                                provider_name = ''
                        else:
                            provider_name = st.selectbox('Provider name', ['-- no providers available --'], key=f'core_provider_name_{visa_type}')
                            provider_name = ''
                        course_start_date = st.date_input('Course start date', key=f'core_course_start_{visa_type}', min_value=date(1900,1,1), max_value=date(2100,12,31))
                        course_end_date = st.date_input('Course end date', key=f'core_course_end_{visa_type}', min_value=date(1900,1,1), max_value=date(2100,12,31))
                        course_expected_duration_months = st.number_input('Course expected duration (months)', min_value=0, step=1, key=f'core_course_expected_months_{visa_type}')
                        time_spent_in_uk_months = st.number_input('Time spent in UK (months)', min_value=0, step=1, key=f'core_time_spent_months_{visa_type}')
                        student_visa_expiry_date = st.date_input('Student visa expiry date', key=f'core_student_visa_expiry_{visa_type}', min_value=date(2000,1,1), max_value=date(2100,12,31))
                        application_date = st.date_input('Application date', key=f'core_application_date_{visa_type}', min_value=date(2000,1,1), max_value=date(2100,12,31))
                        # NEW: Passport fields (identity)
                        country_options = sorted(list(PASSPORT_FORMATS.keys()))
                        passport_issuing_country = st.selectbox('Passport issuing country', country_options + ['Other'], key=f'core_passport_issuing_country_{visa_type}')
                        if passport_issuing_country == 'Other':
                            passport_issuing_country = st.text_input('Please specify passport issuing country', key=f'core_passport_issuing_country_other_{visa_type}')

                        passport_issue_date = st.date_input('Passport issue date', key=f'core_passport_issue_date_{visa_type}', min_value=date(1900,1,1), max_value=date(2100,12,31))
                        passport_expiry_date = st.date_input('Passport expiry date', key=f'core_passport_expiry_date_{visa_type}', min_value=date(1900,1,1), max_value=date(2100,12,31))

                        # NEW: CAS reference (historic)
                        cas_reference = st.text_input('CAS reference used for your completed Student visa', key=f'core_cas_reference_{visa_type}', help='Enter the CAS reference (alphanumeric) you received for the Student visa')

                        # NEW: Proof of valid Student status (BRP or eVisa)
                        immigration_proof_type = st.selectbox('Proof of previous UK immigration status', ['None', 'BRP', 'eVisa'], key=f'core_immigration_proof_type_{visa_type}')
                        brp_expiry_date = None
                        evisa_share_code = ''
                        if immigration_proof_type == 'BRP':
                            brp_expiry_date = st.date_input('BRP expiry date', key=f'core_brp_expiry_date_{visa_type}', min_value=date(1900,1,1), max_value=date(2100,12,31))
                        elif immigration_proof_type == 'eVisa':
                            evisa_share_code = st.text_input('eVisa share code', key=f'core_evisa_share_code_{visa_type}', help='Enter the eVisa share code if available')
                    elif visa_type in ('Skilled Worker', 'Health and Care Worker'):
                        # Objective inputs for Skilled Worker / Health & Care core checks
                        # COS reference (text) instead of yes/no
                        cos_reference = st.text_input('Enter your Certificate of Sponsorship (CoS) reference number', key=f'core_cos_ref_{visa_type}', help='Provide the CoS reference issued by the sponsor')

                        # English evidence type dropdown
                        english_evidence_type = st.selectbox('English language evidence', [
                            'UK degree taught in English',
                            'Approved SELT test',
                            'Nationality exemption',
                            'Not sure'
                        ], key=f'core_english_evidence_{visa_type}')

                        # Occupation code (SOC) and optional job title override
                        # For Skilled Worker provide a dropdown of SOC labels populated from CSV
                        if visa_type == 'Skilled Worker':
                            # Build SOC dropdown labels like '2134 — Programmers and software development professionals'
                            try:
                                soc_df = pd.read_csv('skilled_worker_soc_codes.csv', dtype=str)
                                soc_df['SOC code'] = soc_df['SOC code'].astype(str).str.strip()
                                soc_df['Job type'] = soc_df['Job type'].astype(str).str.strip()
                                # Keep first job type per SOC code
                                soc_map = soc_df.groupby('SOC code')['Job type'].first().to_dict()
                                soc_dropdown = [f"{soc} — {title}" for soc, title in soc_map.items()]
                                soc_dropdown = sorted(soc_dropdown)
                            except Exception:
                                # Fallback to JOB_TITLE_TO_SOC mapping
                                soc_dropdown = sorted([f"{soc} — {jt}" for jt, soc in JOB_TITLE_TO_SOC.items()])

                            soc_choice = st.selectbox('Occupation code (SOC)', ['-- select --'] + soc_dropdown, key=f'core_occupation_code_{visa_type}')
                            if soc_choice and soc_choice != '-- select --':
                                occupation_code = soc_choice.split(' — ')[0].strip()
                            else:
                                occupation_code = ''
                        else:
                            occupation_code = st.text_input('Occupation code (SOC) (optional)', key=f'core_occupation_code_{visa_type}', help='Optional SOC/occupation code, e.g., 2136')
                        if visa_type == 'Health and Care Worker':
                            # For Health & Care worker, allow selecting employer here for NHS/approved check
                            sponsor_names = get_licensed_sponsor_names(limit=500)
                            if sponsor_names:
                                employer_name = st.selectbox('Employer name', ['-- select --'] + sponsor_names, key=f'core_employer_name_{visa_type}', help='Select employer name (used to check NHS/approved status)')
                                if employer_name == '-- select --':
                                    employer_name = ''
                            else:
                                employer_name = st.selectbox('Employer name', ['-- no sponsors available --'], key=f'core_employer_name_{visa_type}')
                                employer_name = ''
                        # Note: salary is collected in Basic and will be used by the rule engine; no salary input here per redesign
                    elif visa_type == 'Standard Visitor':
                        # Purpose of visit (dropdown) replacing free-text planned activities
                        purpose_options = [
                            'Tourism / holiday',
                            'Visit family or friends',
                            'Volunteer (up to 30 days with a registered charity)',
                            'In transit (pass through to another country)',
                            'Business (meetings, interviews)',
                            'Permitted paid engagement / event',
                            'School exchange programme',
                            'Short recreational course (up to 30 days)',
                            'Study / placement / exam',
                            'Academic, senior doctor or dentist',
                            'Medical treatment',
                            'Other (specify)'
                        ]
                        purpose_of_visit = st.selectbox('Purpose of visit', purpose_options, key=f'core_purpose_of_visit_{visa_type}')
                        purpose_other = ''
                        if purpose_of_visit == 'Other (specify)':
                            purpose_other = st.text_input('Please specify purpose of visit', key=f'core_purpose_other_{visa_type}')
                        intended_stay_days = st.number_input('Intended length of stay (days)', min_value=0, step=1, key=f'core_intended_stay_days_{visa_type}')

                        # Guidance: permitted and not permitted activities for Standard Visitor
                        with st.expander('What you can and cannot do as a Standard Visitor'):
                            st.markdown('**You can visit the UK as a Standard Visitor for:**')
                            st.markdown('- tourism (holiday or vacation)')
                            st.markdown('- see your family or friends')
                            st.markdown('- volunteer for up to 30 days with a registered charity')
                            st.markdown('- pass through the UK to another country (in transit)')
                            st.markdown('- certain business activities (attending a meeting or interview)')
                            st.markdown('- certain paid engagements/events as an expert (permitted paid engagement)')
                            st.markdown('- take part in a school exchange programme')
                            st.markdown('- do a recreational course of up to 30 days (e.g., a dance course)')
                            st.markdown('- study, do a placement or take an exam')
                            st.markdown('- attend as an academic, senior doctor or dentist')
                            st.markdown('- for medical reasons')

                            st.markdown('**You cannot:**')
                            st.markdown('- do paid or unpaid work for a UK company or as a self-employed person (unless a permitted paid engagement)')
                            st.markdown('- claim public funds (benefits)')
                            st.markdown('- live in the UK for long periods through frequent or successive visits')
                            st.markdown('- marry or register a civil partnership, or give notice of marriage or civil partnership')

                    submitted = st.form_submit_button('Run Core Check', key=f'btn_run_core_{visa_type}')

                if submitted:
                    data = {}
                    if visa_type == 'Graduate':
                        data = {
                            'provider_name': provider_name,
                            'course_start_date': course_start_date,
                            'course_end_date': course_end_date,
                            'course_expected_duration_months': course_expected_duration_months,
                            'time_spent_in_uk_months': time_spent_in_uk_months,
                            'student_visa_expiry_date': student_visa_expiry_date,
                            'application_date': application_date,

                            'passport_issuing_country': passport_issuing_country,
                            'passport_issue_date': passport_issue_date,
                            'passport_expiry_date': passport_expiry_date,

                            'cas_reference': cas_reference,

                            'immigration_proof_type': immigration_proof_type,
                            'brp_expiry_date': brp_expiry_date,
                            'evisa_share_code': evisa_share_code
                        }
                    elif visa_type == 'Skilled Worker':
                        data = {
                            'cos_reference': cos_reference,
                            'english_evidence_type': english_evidence_type,
                            'occupation_code': occupation_code
                        }
                    elif visa_type == 'Health and Care Worker':
                        data = {
                            'cos_reference': cos_reference,
                            'english_evidence_type': english_evidence_type,
                            'occupation_code': occupation_code,
                            'employer_name': employer_name
                        }
                    elif visa_type == 'Standard Visitor':
                        # store purpose_of_visit (use other if specified)
                        pov = purpose_other if (purpose_of_visit == 'Other (specify)') else purpose_of_visit
                        data = {
                            'purpose_of_visit': pov,
                            'intended_stay_days': intended_stay_days
                        }

                    st.session_state.elig_form.update(data)
                    result = evaluator('core', st.session_state.elig_form)
                    st.session_state.elig_result = result

                    # For Skilled Worker, auto-finalize by moving to the Detailed finalisation step
                    if visa_type == 'Skilled Worker':
                        st.session_state.elig_step = 'detailed'

                if st.session_state.get('elig_result'):
                    result = st.session_state.elig_result
                    if result.get('failed_rules'):
                        st.error('❌ Core checks failed')
                        retrieved = retrieve_with_rag(result.get('failed_rules', []))
                        st.session_state.elig_retrieved = retrieved
                        if st.session_state.get('elig_explanation'):
                            expl = st.session_state.elig_explanation
                        else:
                            expl = None
                            if st.button('Request detailed LLM explanation', key=f'btn_request_llm_core_{visa_type}'):
                                with st.spinner('Generating LLM explanation...'):
                                    expl = llm_explain({'rule_results': result}, retrieved)
                                    st.session_state.elig_explanation = expl
                        st.markdown('### LLM Explanation')
                        if expl:
                            st.markdown(f"**Decision:** {expl.get('decision')}")
                            if expl.get('per_rule'):
                                st.markdown('#### Per-rule explanations')
                                for pr in expl.get('per_rule'):
                                    st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                    cit = pr.get('citation') or {}
                                    if cit:
                                        st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                        st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                        if cit.get('section'):
                                            st.markdown(f"  - Section: {cit.get('section')}")
                            if expl.get('recommendations'):
                                st.markdown('#### Recommendations')
                                for rec in expl.get('recommendations'):
                                    st.markdown(f"- {rec}")
                            if not expl.get('per_rule') and not expl.get('recommendations'):
                                st.markdown(f"**Summary:** {expl.get('summary')}")
                            decision = expl.get('decision', '')
                            if decision and 'not' in decision.lower():
                                failed = result.get('failed_rules', [])
                                reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                                if reasons:
                                    one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                                    st.markdown(f"**Quick reason:** {one_line}")

                        if retrieved:
                            st.markdown('### Supporting citations')
                            for c in retrieved:
                                st.markdown(f"**Document:** {c.get('doc','Unknown')}")
                                st.markdown(f"**Page:** {c.get('page','N/A')}")
                                st.markdown(f"**Section/Paragraph:** {c.get('section','N/A')}")
                    else:
                        st.success('✅ Core checks passed')
                        st.session_state.elig_retrieved = []
                        st.session_state.elig_explanation = None
                        passed = result.get('passed_rules', [])
                        if passed:
                            st.markdown('**Passed checks:**')
                            for r in passed:
                                st.markdown(f'- ✅ {r}')

                        # For Skilled Worker we auto-finalize after Core (no manual proceed button). For Health & Care we do not show Core/Detailed.
                        if visa_type not in ('Health and Care Worker', 'Skilled Worker', 'Student'):
                            if st.button('Proceed to Detailed check', key=f'btn_proceed_detailed_{visa_type}'):
                                st.session_state.elig_step = 'detailed'

                    if st.button('Back to Basic', key=f'btn_back_to_basic_{visa_type}'):
                        st.session_state.elig_step = 'basic'

            # --- DETAILED ---
            elif st.session_state.elig_step == 'detailed':
                # DETAILED step: for Graduate and Skilled Worker we do not render a detailed input form.
                # Instead the Detailed step acts as a finalisation step that re-runs Core evaluation
                # and displays the final result (no extra user inputs required).
                if st.session_state.elig_step == 'detailed':
                    if visa_type == 'Graduate':
                        # For Graduate visa: do not render a Detailed input form.
                        # Treat the Detailed step as a finalisation step that re-runs
                        # the Core evaluation and displays the final result (no extra inputs).
                        result = evaluator('core', st.session_state.elig_form)
                        st.session_state.elig_result = result

                        if result.get('failed_rules'):
                            st.error('❌ Final checks failed')
                            retrieved = retrieve_with_rag(result.get('failed_rules', []))
                            st.session_state.elig_retrieved = retrieved
                            if st.session_state.get('elig_explanation'):
                                expl = st.session_state.elig_explanation
                            else:
                                expl = None
                                if st.button('Request detailed LLM explanation', key=f'btn_request_llm_final_{visa_type}'):
                                    with st.spinner('Generating LLM explanation...'):
                                        expl = llm_explain({'rule_results': result}, retrieved)
                                        st.session_state.elig_explanation = expl
                        else:
                            st.success('✅ You meet the configured eligibility checks for this visa type')
                            st.session_state.elig_retrieved = []
                            st.session_state.elig_explanation = None

                        st.markdown('---')
                        st.markdown('### Rule results')
                        for r in result.get('passed_rules', []):
                            st.markdown(f"- ✅ {r}")
                        for r in result.get('failed_rules', []):
                            st.markdown(f"- ❌ {r}")

                        if st.session_state.get('elig_explanation'):
                            st.markdown('### Explanation')
                            expl = st.session_state.elig_explanation
                            st.markdown(f"**Decision:** {expl.get('decision')}")
                            st.markdown(f"**Summary:** {expl.get('summary')}")
                            decision = expl.get('decision', '')
                            if decision and 'not' in decision.lower():
                                failed = result.get('failed_rules', [])
                                reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                                if reasons:
                                    one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                                    st.markdown(f"**Quick reason:** {one_line}")

                            if expl.get('per_rule'):
                                st.markdown('#### Per-rule explanations')
                                for pr in expl.get('per_rule'):
                                    st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                    cit = pr.get('citation') or {}
                                    st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                    st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                    st.markdown(f"  - Section: {cit.get('section','')}")
                                    # Synthesize a short, user-friendly explanation for common failures
                                    failed_rules = result.get('failed_rules', [])
                                    friendly_lines = []
                                    if any(r in ('NO_CAS', 'CAS_PRESENT', 'CAS_REFERENCE_MISSING') for r in failed_rules):
                                        friendly_lines.append(
                                            "A CAS (Confirmation of Acceptance for Studies) is an official reference issued by a licensed education provider.\n"
                                            "It proves you have an offer and a place on a specific course, and includes course and sponsor details.\n"
                                            "The Student route requires a CAS because it shows the Home Office that a licensed sponsor is taking responsibility for your study.\n"
                                        )
                                    if any(r in ('FUNDS_INSUFFICIENT', 'FUNDS_NOT_HELD_28_DAYS', 'FUNDS_28') for r in failed_rules):
                                        friendly_lines.append(
                                            "Financial evidence: you must show you have enough money to cover course fees and living costs, and that the funds have been held for 28 consecutive days.\n"
                                        )
                                    if any(r in ('PROVIDER_LICENSED',) for r in failed_rules):
                                        friendly_lines.append(
                                            "Provider must be licensed: your education provider needs a Home Office sponsor licence to issue a valid CAS.\n"
                                        )
                                    if friendly_lines:
                                        st.markdown('---')
                                        st.markdown('### Plain-language help')
                                        for para in friendly_lines:
                                            st.markdown(para.replace('\\n', '  \\n'))
                            if expl.get('recommendations'):
                                st.markdown('#### Recommendations')
                                for rec in expl.get('recommendations'):
                                    st.markdown(f"- {rec}")
                    elif visa_type == 'Skilled Worker':
                        # Re-run the core evaluation to compute final determination
                        result = evaluator('core', st.session_state.elig_form)
                        st.session_state.elig_result = result

                        if result.get('failed_rules'):
                            st.error('❌ Final checks failed')
                            retrieved = retrieve_with_rag(result.get('failed_rules', []))
                            st.session_state.elig_retrieved = retrieved
                            if st.session_state.get('elig_explanation'):
                                expl = st.session_state.elig_explanation
                            else:
                                expl = None
                                if st.button('Request detailed LLM explanation', key=f'btn_request_llm_final_{visa_type}'):
                                    with st.spinner('Generating LLM explanation...'):
                                        expl = llm_explain({'rule_results': result}, retrieved)
                                        st.session_state.elig_explanation = expl
                        else:
                            st.success('✅ You meet the configured eligibility checks for this visa type')
                            st.session_state.elig_retrieved = []
                            st.session_state.elig_explanation = None

                        st.markdown('---')
                        st.markdown('### Rule results')
                        for r in result.get('passed_rules', []):
                            st.markdown(f"- ✅ {r}")
                        for r in result.get('failed_rules', []):
                            st.markdown(f"- ❌ {r}")

                        if st.session_state.get('elig_retrieved'):
                            st.markdown('### Supporting citations')
                            for c in st.session_state.elig_retrieved:
                                st.markdown('**Document:** ' + str(c.get('doc', 'Unknown')))
                                st.markdown('**Page:** ' + str(c.get('page', 'N/A')))
                                st.markdown('**Section:** ' + str(c.get('section', '')))

                        if st.button('Back to Core', key=f'btn_back_to_core_{visa_type}'):
                            st.session_state.elig_step = 'core'
                    else:
                        # For other visas (e.g., Standard Visitor), render the existing detailed form
                        with st.form(f'detailed_form_{visa_type}'):
                            st.write('#### Detailed checks(optional)')
                            st.success('###### you are eligible for a standard visitor visa, you can provide additional optional information if you wish to strengthen your application')
                            if visa_type in ('Health and Care Worker',):
                                # Health & Care Worker no longer has Core/Detailed per requirement; this branch is left
                                # for backwards compatibility but will normally not be reached.
                                pass
                            elif visa_type == 'Standard Visitor':
                                accommodation_details = st.text_area('Accommodation details', key=f'detailed_accom_{visa_type}')
                                travel_history = st.text_area('Travel history (recent)', key=f'detailed_travel_history_{visa_type}')
                                ties_to_home_country = st.text_input('Ties to home country (family/employment)', key=f'detailed_ties_{visa_type}')

                            submitted = st.form_submit_button('Run Final Check')

                        if submitted:
                            data = {}
                            if visa_type == 'Standard Visitor':
                                data = {
                                    'accommodation_details': accommodation_details,
                                    'travel_history': travel_history,
                                    'ties_to_home_country': ties_to_home_country
                                }

                            st.session_state.elig_form.update(data)
                            result = evaluator('detailed', st.session_state.elig_form)
                            st.session_state.elig_result = result

                            if result.get('failed_rules'):
                                st.error('❌ Final checks failed')
                                retrieved = retrieve_with_rag(result.get('failed_rules', []))
                                st.session_state.elig_retrieved = retrieved
                                expl = llm_explain({'rule_results': result}, retrieved)
                                st.session_state.elig_explanation = expl
                            else:
                                st.success('✅ You meet the configured eligibility checks for this visa type')
                                st.session_state.elig_retrieved = []
                                st.session_state.elig_explanation = None

                            st.markdown('---')
                            st.markdown('### Rule results')
                            for r in result.get('passed_rules', []):
                                st.markdown(f"- ✅ {r}")
                            for r in result.get('failed_rules', []):
                                st.markdown(f"- ❌ {r}")

                            if st.session_state.get('elig_explanation'):
                                st.markdown('### Explanation')
                                expl = st.session_state.elig_explanation
                                st.markdown(f"**Decision:** {expl.get('decision')}")
                                # Show per-rule explanations and recommendations only
                                if expl.get('per_rule'):
                                    st.markdown('#### Per-rule explanations')
                                    for pr in expl.get('per_rule'):
                                        st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                        cit = pr.get('citation') or {}
                                        if cit:
                                            st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                            st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                            if cit.get('section'):
                                                st.markdown(f"  - Section: {cit.get('section')}")
                                if expl.get('recommendations'):
                                    st.markdown('#### Recommendations')
                                    for rec in expl.get('recommendations'):
                                        st.markdown(f"- {rec}")
                                if not expl.get('per_rule') and not expl.get('recommendations'):
                                    st.markdown(f"**Summary:** {expl.get('summary')}")
                                decision = expl.get('decision', '')
                                if decision and 'not' in decision.lower():
                                    failed = result.get('failed_rules', [])
                                    reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                                    if reasons:
                                        one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                                        st.markdown(f"**Quick reason:** {one_line}")

                                if expl.get('per_rule'):
                                    st.markdown('#### Per-rule explanations')
                                    for pr in expl.get('per_rule'):
                                        st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                        cit = pr.get('citation') or {}
                                        st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                        st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                        st.markdown(f"  - Section: {cit.get('section','')}")
                                if expl.get('recommendations'):
                                    st.markdown('#### Recommendations')
                                    for rec in expl.get('recommendations'):
                                        st.markdown(f"- {rec}")

                                    # Also provide a short, user-friendly plain-language explanation
                                    # synthesised from the failed rules and citations (helpful for end users).
                                    failed_rules = result.get('failed_rules', [])
                                    if failed_rules:
                                        friendly_lines = []

                                        # If CAS-related failures present, explain what a CAS is and why it's needed
                                        if any(r in ('NO_CAS', 'CAS_PRESENT', 'CAS_REFERENCE_MISSING') for r in failed_rules):
                                            friendly_lines.append(
                                                "What is a CAS (Confirmation of Acceptance for Studies)?\n"
                                                "A CAS is an official reference number issued by a licensed education provider (sponsor) \n"
                                                "to confirm they've offered you a place on a specific course. It includes course details, \n"
                                                "the course start date and the sponsor's details.\n"
                                            )
                                            friendly_lines.append(
                                                "Why do you need it?\n"
                                                "For the Student visa route, the UK Home Office requires evidence you have a confirmed \n"
                                                "sponsored place. A valid CAS proves the provider has offered you that place — without it \n"
                                                "your application cannot be accepted.\n"
                                            )
                                            friendly_lines.append(
                                                "How to fix it (next steps):\n"
                                                "1. Contact the education provider and confirm they will issue a CAS once you meet their \n"
                                                "   conditions (e.g., offer acceptance, payment of deposit).\n"
                                                "2. Make sure the provider holds a valid sponsor licence.\n"
                                                "3. When you receive the CAS, check the reference and dates carefully and include it in your application.\n"
                                            )

                                        # Funds-related guidance
                                        if any(r in ('FUNDS_INSUFFICIENT', 'FUNDS_NOT_HELD_28_DAYS', 'FUNDS_28') for r in failed_rules):
                                            friendly_lines.append(
                                                "Financial requirements — plain language:\n"
                                                "You must show you have enough money to pay your tuition fees and living costs for the period \n"
                                                "required by the visa rules, and that the funds have been in your account for at least 28 consecutive days. \n"
                                                "If your bank statement or evidence doesn't meet those requirements, gather clearer evidence or top up balances.\n"
                                            )

                                        # Provider licence guidance
                                        if any(r in ('PROVIDER_LICENSED',) for r in failed_rules):
                                            friendly_lines.append(
                                                "Provider licensing:\n"
                                                "Your course must be offered by a provider that holds a UK sponsor licence. If the provider is not \n"
                                                "licensed, they cannot issue a valid CAS. Check the provider's sponsor status on the official list.\n"
                                            )

                                        if friendly_lines:
                                            st.markdown('---')
                                            st.markdown('### Plain-language help')

                            if st.session_state.get('elig_retrieved'):
                                            for para in friendly_lines:
                                                # render each paragraph as a markdown block for readability
                                                st.markdown(para.replace('\\n', '  \\n'))

                            if st.session_state.get('elig_retrieved'):
                                st.markdown('### Supporting citations')
                                for c in st.session_state.elig_retrieved:
                                    st.markdown('**Document:** ' + str(c.get('doc', 'Unknown')))
                                    st.markdown('**Page:** ' + str(c.get('page', 'N/A')))
                                    st.markdown('**Section:** ' + str(c.get('section', '')))

                            if st.button('Back to Core', key=f'btn_back_to_core_{visa_type}'):
                                st.session_state.elig_step = 'core'

            # stop further (Student) UI from rendering
            st.stop()

        # --- STEP 1: BASIC CHECK ---
        if st.session_state.elig_step == 'basic':
            with st.form(f'basic_form_{visa_type}'):
                st.write('#### Basic information')
                # Render inputs in a responsive card-grid while preserving original keys and behaviour
                st.markdown('<div class="card-grid">', unsafe_allow_html=True)
                # Column-like cards using markup; inputs keep the same session keys

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Personal</div>', unsafe_allow_html=True)
                st.markdown("**Date of birth***", unsafe_allow_html=True)
                dob = st.date_input('', key='basic_date_of_birth', min_value=date(1900, 1, 1), max_value=date(2100, 12, 31), help='Required. Format: YYYY-MM-DD')
                field_info_toggle('dob', 'Enter your date of birth. Format: YYYY-MM-DD. Use the calendar to pick year; range is 1900-2100.')

                st.markdown("**Nationality***", unsafe_allow_html=True)
                country_options = sorted(list(PASSPORT_FORMATS.keys())) + ['Other']
                nationality = st.selectbox('', country_options, index=0, key='basic_nationality', help='Required. Choose your nationality from the list or select Other to type.')
                field_info_toggle('nationality', 'Select your country of nationality from the dropdown. If Other, specify in the box that appears.')
                other_nationality = None
                if nationality == 'Other':
                    other_nationality = st.text_input('Please specify nationality', key='basic_nationality_other', help='Type your nationality, e.g., Brazil')
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Passport & Location</div>', unsafe_allow_html=True)
                country_options = sorted(list(PASSPORT_FORMATS.keys())) + ['Other']
                passport_issuing_country = st.selectbox('Passport issuing country', country_options, key='basic_passport_issuing_country')
                if passport_issuing_country == 'Other':
                    passport_issuing_country = st.text_input('Please specify issuing country', key='basic_passport_issuing_country_other')

                passport_issue_date = st.date_input('Passport issue date', key='basic_passport_issue_date', min_value=date(1900,1,1), max_value=date(2100,12,31))
                passport_expiry_date = st.date_input('Passport expiry date', key='basic_passport_expiry_date', min_value=date(1900,1,1), max_value=date(2100,12,31))

                currently_in_uk = st.selectbox('Where are you applying from?', ['Inside the UK', 'Outside the UK'], key='basic_current_location')
                currently_in_uk_bool = (currently_in_uk == 'Inside the UK')
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Health & Routing</div>', unsafe_allow_html=True)
                field_info_toggle('cas', 'A CAS (Confirmation of Acceptance for Studies) is issued by the licensed sponsor. It usually looks like a reference number and is required for Student route applications.')

                tb_required_set = fetch_tb_required_countries()
                tb_test_date = None
                submitted = None
                nat_value = (other_nationality.strip() if other_nationality else nationality)
                nat_norm = nat_value.strip().lower() if nat_value else ''
                tb_required_norm = set([c.strip().lower() for c in tb_required_set])
                if nat_norm and nat_norm in tb_required_norm:
                    st.warning('Applicants from this country usually need a TB test to apply for a UK visa.')
                    tb_test_date = st.date_input('TB test date (if taken)', key='basic_tb_test_date', min_value=date(1900,1,1), max_value=date(2100,12,31), help='Provide the TB test date if you already had one.')

                # submit button (keeps identical key)
                submitted = st.form_submit_button('Run Basic Check', key=f'btn_run_basic_{visa_type}')
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            if submitted:
                data = {
                    'date_of_birth': dob,
                    'nationality': nat_value,
                    'currently_in_uk': currently_in_uk_bool,
                    'passport_issuing_country': passport_issuing_country,
                    'passport_issue_date': passport_issue_date,
                    'passport_expiry_date': passport_expiry_date,
                    'tb_test_date': tb_test_date,
                    'application_date': date.today()
                }
                st.session_state.elig_form.update(data)

                if visa_type == 'Student':
                    result = evaluate_stub('basic', st.session_state.elig_form)
                else:
                    result = evaluate_stub('basic', st.session_state.elig_form)

                st.session_state.elig_result = result

            # preserve existing rendering behavior below
            if st.session_state.get('elig_result'):
                result = st.session_state.elig_result
                if result.get('failed_rules'):
                    st.error('❌ Basic checks failed')
                    failed = result.get('failed_rules', [])
                    reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                    if reasons:
                        one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                        st.markdown(f"**Quick reason:** {one_line}")
                    retrieved = retrieve_with_rag(result.get('failed_rules', []))
                    st.session_state.elig_retrieved = retrieved
                    expl = llm_explain({'rule_results': result}, retrieved)
                    st.session_state.elig_explanation = expl

                    st.markdown('### LLM Explanation')
                    if expl:
                        st.markdown(f"**Decision:** {expl.get('decision')}")
                        if expl.get('per_rule'):
                            st.markdown('#### Per-rule explanations')
                            for pr in expl.get('per_rule'):
                                st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                cit = pr.get('citation') or {}
                                if cit:
                                    st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                    st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                    if cit.get('section'):
                                        st.markdown(f"  - Section: {cit.get('section')}")
                        if expl.get('recommendations'):
                            st.markdown('#### Recommendations')
                            for rec in expl.get('recommendations'):
                                st.markdown(f"- {rec}")
                        if not expl.get('per_rule') and not expl.get('recommendations'):
                            st.markdown(f"**Summary:** {expl.get('summary')}")
                        decision = expl.get('decision', '')
                        if decision and 'not' in decision.lower():
                            failed = result.get('failed_rules', [])
                            reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                            if reasons:
                                one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                                st.markdown(f"**Quick reason:** {one_line}")

                        if expl.get('per_rule'):
                            st.markdown('#### Per-rule explanations')
                            for pr in expl.get('per_rule'):
                                st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                cit = pr.get('citation') or {}
                                st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                st.markdown(f"  - Section: {cit.get('section','')}")
                        if expl.get('recommendations'):
                            st.markdown('#### Recommendations')
                            for rec in expl.get('recommendations'):
                                st.markdown(f"- {rec}")

                    # Then show RAG citations without quoting the full chunk
                    if retrieved:
                        st.markdown('### Supporting citations (check source authenticity)')
                        for c in retrieved:
                            st.markdown(f"**Document:** {c.get('doc','Unknown')}")
                            st.markdown(f"**Page:** {c.get('page','N/A')}")
                            st.markdown(f"**Section/Paragraph:** {c.get('section','N/A')}")
                else:
                    st.success('✅ Basic checks passed')
                    st.session_state.elig_retrieved = []
                    st.session_state.elig_explanation = None
                    passed = result.get('passed_rules', [])
                    if passed:
                        st.markdown('**Passed checks:**')
                        for r in passed:
                            st.markdown(f'- ✅ {r}')

                    if st.button('Proceed to Core check', key='btn_proceed_core'):
                        st.session_state.elig_step = 'core'

        # --- STEP 2: CORE CHECK ---
        elif st.session_state.elig_step == 'core':
            with st.form(f'core_form_{visa_type}'):
                st.write('#### Core checks')
                if visa_type == 'Student':
                    # Student core: structured factual inputs (providers from CSV)
                    student_providers = get_licensed_student_provider_names(limit=500)
                    if student_providers:
                        provider = st.selectbox('Education provider', ['-- select --'] + student_providers, key='core_provider_name')
                        if provider == '-- select --':
                            provider = ''
                    else:
                        provider = st.selectbox('Education provider', ['-- no providers available --'], key='core_provider_name')
                        provider = ''

                    cas_number = st.text_input('CAS reference number', key='core_cas_number')
                    course_level = st.selectbox('Course level', ["Bachelor's", "Master's", 'PhD', 'Foundation', 'Language'], key='core_course_level')
                    course_mode = st.selectbox('Study mode', ['Full-time', 'Part-time'], key='core_course_mode')
                    course_start = st.date_input('Course start date', key='core_course_start', min_value=date(1900,1,1), max_value=date(2100,12,31))
                    course_end = st.date_input('Course end date', key='core_course_end', min_value=date(1900,1,1), max_value=date(2100,12,31))
                    funds_amount = st.number_input('Funds available (£)', min_value=0, step=50, key='core_funds_amount')
                    funds_held_since = st.date_input('Funds held since', key='core_funds_held_since', min_value=date(1900,1,1), max_value=date(2100,12,31))
                    funds_source = st.selectbox('Source of funds', ['Bank account', 'Savings', 'Scholarship', 'Loan', 'Family support', 'Other'], key='core_funds_source')
                    funds_source_other = ''
                    if funds_source == 'Other':
                        funds_source_other = st.text_input('Please describe the source of funds', key='core_funds_source_other')
                    evidence_date = st.date_input('Evidence date (statement date)', key='core_evidence_date', min_value=date(1900,1,1), max_value=date(2100,12,31))
                    # Additional factual inputs required for financial calculation
                    course_fee = st.number_input('Course fee (GBP) (leave 0 if unknown)', min_value=0, step=50, key='core_course_fee')
                    has_dependants = st.checkbox('Do you have dependants?', key='core_has_dependants')
                    num_dependants = 0
                    if has_dependants:
                        num_dependants = st.number_input('Number of dependants', min_value=1, step=1, key='core_num_dependants')
                    # Study location to decide London / outside London rates
                    in_london = st.selectbox('Study location', ['Outside London', 'London'], key='core_in_london')
                    english_evidence = st.selectbox('English evidence', ['UK degree', 'Approved SELT', 'Nationality exemption', 'Not sure'], key='core_english_evidence')
                    selt_cefr = None
                    if english_evidence == 'Approved SELT':
                        selt_cefr = st.selectbox('If you have a SELT, select CEFR level', ['', 'B1', 'B2', 'C1'], key='core_selt_cefr')

                else:
                    # Non-student core (Graduate handled above in non-student block)
                    provider = None
                    cas_number = None
                    course_level = None
                    course_mode = None
                    course_start = None
                    course_end = None
                    funds_amount = None
                    funds_held_since = None
                    english_evidence = None
                    selt_cefr = None

                submitted = st.form_submit_button('Run Core Check')

            if submitted:
                # Build data and compute booleans expected by the rule engine
                data = {}
                if visa_type == 'Student':
                    data = {
                        'provider_name': provider,
                        'provider_is_licensed': bool(provider),
                        'cas_number': cas_number,
                        'has_cas': bool(cas_number and cas_number.strip()),
                        'course_level': course_level,
                        'course_full_time': (course_mode == 'Full-time') if course_mode else False,
                        'course_start_date': course_start,
                        'course_end_date': course_end,
                            'funds_amount': funds_amount,
                            'funds_held_since': funds_held_since,
                            'course_fee': course_fee,
                            'num_dependants': num_dependants,
                            'in_london': True if in_london == 'London' else False,
                        'funds_held_28_days': False,
                            'funds_source': (funds_source_other if funds_source == 'Other' else funds_source),
                            'evidence_date': evidence_date,
                        'english_exempt_or_test': english_evidence in ('UK degree', 'Nationality exemption'),
                        'selt_cefr_level': selt_cefr if selt_cefr else None
                    }
                    # compute funds 28-day boolean defensively
                    try:
                        if funds_held_since and (date.today() - funds_held_since).days >= 28 and funds_amount and funds_amount > 0:
                            data['funds_held_28_days'] = True
                    except Exception:
                        data['funds_held_28_days'] = False

                    st.session_state.elig_form.update(data)
                    # Run student rules
                    result = evaluate_student('core', st.session_state.elig_form)

                    # Financial checks integration: compute requirement and validate evidence
                    fin_req = check_financial_requirement({
                        'nationality': st.session_state.elig_form.get('nationality'),
                        'time_spent_in_uk_months': st.session_state.elig_form.get('time_spent_in_uk_months')
                    })
                    fin_calc = calculate_required_funds({
                        'course_months': ((st.session_state.elig_form.get('course_expected_duration_months') or 0) or 0),
                        'in_london': st.session_state.elig_form.get('in_london'),
                        'course_fee': st.session_state.elig_form.get('course_fee') or 0,
                        'num_dependants': st.session_state.elig_form.get('num_dependants') or 0
                    })
                    fin_val = validate_financial_evidence({
                        'funds_amount': st.session_state.elig_form.get('funds_amount'),
                        'funds_held_since': st.session_state.elig_form.get('funds_held_since'),
                        'evidence_date': st.session_state.elig_form.get('funds_held_since'),
                        'application_date': st.session_state.elig_form.get('application_date') or date.today(),
                        'funds_source': st.session_state.elig_form.get('funds_source') or ''
                    })

                    # Attach financial results to elig_result for UI rendering
                    result['financial'] = {
                        'requirement': fin_req,
                        'calculation': fin_calc,
                        'validation': fin_val
                    }
                    # Evaluate financial failure conditions and augment failed_rules
                    failed_rules = list(result.get('failed_rules', []))

                    applicant_funds = float(st.session_state.elig_form.get('funds_amount') or 0)
                    required_total = float(fin_calc.get('total_required', 0))

                    # Funds insufficiency
                    if fin_req.get('required'):
                        if applicant_funds < required_total:
                            failed_rules.append('FUNDS_INSUFFICIENT')

                        # Merge any validation failures
                        for fr in fin_val.get('fail_reasons', []):
                            # use the same codes returned by validate_financial_evidence
                            failed_rules.append(fr)

                        # Upload requirement: if nationality not exempt and upload not provided
                        uploaded = st.session_state.get('core_financial_upload')
                        if not fin_req.get('exempt_nationality') and not uploaded:
                            failed_rules.append('UPLOAD_MISSING')

                    # Deduplicate and set back
                    result['failed_rules'] = list(dict.fromkeys(failed_rules))
                else:
                    st.session_state.elig_form.update(data)
                    result = evaluate_stub('core', st.session_state.elig_form)

                st.session_state.elig_result = result

            # Render stored result for Core step
            if st.session_state.get('elig_result'):
                result = st.session_state.elig_result
                if result.get('failed_rules'):
                    st.error('❌ Core checks failed')
                    retrieved = retrieve_with_rag(result.get('failed_rules', []))
                    st.session_state.elig_retrieved = retrieved
                    expl = llm_explain({'rule_results': result}, retrieved)
                    st.session_state.elig_explanation = expl

                    # Show LLM explanation first
                    st.markdown('### LLM Explanation')
                    if expl:
                        st.markdown(f"**Decision:** {expl.get('decision')}")
                        if expl.get('per_rule'):
                            st.markdown('#### Per-rule explanations')
                            for pr in expl.get('per_rule'):
                                st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                cit = pr.get('citation') or {}
                                if cit:
                                    st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                    st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                    if cit.get('section'):
                                        st.markdown(f"  - Section: {cit.get('section')}")
                        if expl.get('recommendations'):
                            st.markdown('#### Recommendations')
                            for rec in expl.get('recommendations'):
                                st.markdown(f"- {rec}")
                        if not expl.get('per_rule') and not expl.get('recommendations'):
                            st.markdown(f"**Summary:** {expl.get('summary')}")
                        decision = expl.get('decision', '')
                        if decision and 'not' in decision.lower():
                            failed = result.get('failed_rules', [])
                            reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                            if reasons:
                                one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                                st.markdown(f"**Quick reason:** {one_line}")

                        if expl.get('per_rule'):
                            st.markdown('#### Per-rule explanations')
                            for pr in expl.get('per_rule'):
                                st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                cit = pr.get('citation') or {}
                                st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                st.markdown(f"  - Section: {cit.get('section','')}")
                        if expl.get('recommendations'):
                            st.markdown('#### Recommendations')
                            for rec in expl.get('recommendations'):
                                st.markdown(f"- {rec}")

                    # Then show RAG citations without quoting the full chunk
                    if retrieved:
                        st.markdown('### Supporting citations (check source authenticity)')
                        for c in retrieved:
                            st.markdown(f"**Document:** {c.get('doc','Unknown')}")
                            st.markdown(f"**Page:** {c.get('page','N/A')}")
                            st.markdown(f"**Section/Paragraph:** {c.get('section','N/A')}")
                else:
                    st.success('✅ Core checks passed')
                    st.session_state.elig_retrieved = []
                    st.session_state.elig_explanation = None
                    passed = result.get('passed_rules', [])
                    # Show financial calculation and comparison when present
                    fin = result.get('financial') or {}
                    if fin:
                        fr = fin.get('requirement', {})
                        fc = fin.get('calculation', {})
                        fv = fin.get('validation', {})

                        st.markdown('---')
                        st.markdown('### Financial requirement')
                        if fr.get('required'):
                            st.warning('Financial evidence is required for this application')
                            if fr.get('reasons'):
                                st.markdown('**Reasons:** ' + ', '.join(fr.get('reasons', [])))
                        else:
                            st.success('Financial evidence is NOT required')
                            if fr.get('reasons'):
                                st.markdown('**Exemption reasons:** ' + ', '.join(fr.get('reasons', [])))

                        st.markdown('**Required funds breakdown:**')
                        st.markdown(f"- Course fee: £{fc.get('course_fee', 0):,.2f}")
                        st.markdown(f"- Living cost: £{fc.get('living_cost', 0):,.2f}")
                        st.markdown(f"- Dependants cost: £{fc.get('dependant_cost', 0):,.2f}")
                        st.markdown(f"- **Total required:** £{fc.get('total_required', 0):,.2f}")

                        applicant_funds = st.session_state.elig_form.get('funds_amount') or 0
                        st.markdown(f"**Applicant declared funds:** £{applicant_funds:,.2f}")
                        if applicant_funds >= fc.get('total_required', 0):
                            st.success('Declared funds meet or exceed the required amount')
                        else:
                            st.error('Declared funds are below the required amount')

                        # Show validation summary
                        st.markdown('**Financial evidence validation:**')
                        if fv.get('valid'):
                            st.success('Evidence appears valid (dates and source checks pass)')
                        else:
                            st.error('Evidence validation failed: ' + ', '.join(fv.get('fail_reasons', [])))

                        # Show upload field only if required and nationality is not exempt
                        if fr.get('required') and not fr.get('exempt_nationality'):
                            st.markdown('Please upload financial evidence (bank statement, sponsor letter)')
                            uploaded = st.file_uploader('Upload financial evidence', type=['pdf', 'png', 'jpg', 'jpeg'], key='core_financial_upload')
                            if uploaded:
                                st.markdown('File uploaded: ' + uploaded.name)
                    if passed:
                        st.markdown('**Passed checks:**')
                        for r in passed:
                            st.markdown(f'- ✅ {r}')

                    if st.button('Proceed to Detailed check', key='btn_proceed_detailed'):
                        st.session_state.elig_step = 'detailed'

                # Back button to previous step
                if st.button('Back to Basic', key='btn_back_to_basic_from_core'):
                    st.session_state.elig_step = 'basic'

        # --- STEP 3: DETAILED CHECK ---
        elif st.session_state.elig_step == 'detailed':
            # For Student visa we no longer render a detailed input form. Treat Detailed
            # as the finalisation step: re-run the Student core evaluation and display
            # the final result (no extra user inputs required).
            if visa_type == 'Student':
                result = evaluate_student('core', st.session_state.elig_form)
                st.session_state.elig_result = result

                if result.get('failed_rules'):
                    st.error('❌ Final checks failed')
                    retrieved = retrieve_with_rag(result.get('failed_rules', []))
                    st.session_state.elig_retrieved = retrieved
                    expl = llm_explain({'rule_results': result}, retrieved)
                    st.session_state.elig_explanation = expl
                else:
                    st.success('✅ You meet the configured eligibility checks for this visa type')
                    st.session_state.elig_retrieved = []
                    st.session_state.elig_explanation = None

                # Show per-rule results (same display as original Detailed flow)
                st.markdown('---')
                st.markdown('### Rule results')
                for r in result.get('passed_rules', []):
                    st.markdown(f"- ✅ {r}")
                for r in result.get('failed_rules', []):
                    st.markdown(f"- ❌ {r}")

                if st.session_state.get('elig_explanation'):
                    st.markdown('### Explanation')
                    expl = st.session_state.elig_explanation
                    st.markdown(f"**Decision:** {expl.get('decision')}")
                    st.markdown(f"**Summary:** {expl.get('summary')}")
                    decision = expl.get('decision', '')
                    if decision and 'not' in decision.lower():
                        failed = result.get('failed_rules', [])
                        reasons = [RULE_SHORT_REASON.get(r, f'failed requirement {r}') for r in failed]
                        if reasons:
                            one_line = 'Not eligible because ' + '; '.join(reasons) + '.'
                            st.markdown(f"**Quick reason:** {one_line}")

                    if expl.get('per_rule'):
                        st.markdown('#### Per-rule explanations')
                        for pr in expl.get('per_rule'):
                            st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                            cit = pr.get('citation') or {}
                            st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                            st.markdown(f"  - Page: {cit.get('page','N/A')}")
                            st.markdown(f"  - Section: {cit.get('section','')}")
                    if expl.get('recommendations'):
                        st.markdown('#### Recommendations')
                        for rec in expl.get('recommendations'):
                            st.markdown(f"- {rec}")

                if st.session_state.get('elig_retrieved'):
                    st.markdown('### Supporting citations')
                    for c in st.session_state.elig_retrieved:
                        st.markdown('**Document:** ' + str(c.get('doc', 'Unknown')))
                        st.markdown('**Page:** ' + str(c.get('page', 'N/A')))
                        st.markdown('**Section:** ' + str(c.get('section', '')))
                if st.button('Back to Core', key='btn_back_to_core_from_detailed'):
                    st.session_state.elig_step = 'core'


    # --- New tab: eligibility-final (compact common + per-visa eligibility form) ---
    with tab5:
        st.markdown("### eligibility-final")

        # Keep a compact session storage for the form and results
        if 'elig_final_form' not in st.session_state:
            st.session_state.elig_final_form = {}
        if 'elig_final_result' not in st.session_state:
            st.session_state.elig_final_result = None
        if 'elig_final_retrieved' not in st.session_state:
            st.session_state.elig_final_retrieved = []
        if 'elig_final_explanation' not in st.session_state:
            st.session_state.elig_final_explanation = None

        # Simple country list (kept small to avoid depending on PASSPORT_FORMATS scope)
        country_options = ["United Kingdom", "India", "United States", "Canada", "Australia", "Other"]

        # Step 1: common details form. User fills these and clicks Continue.
        with st.form(key='elig_final_common_form'):
            st.markdown('#### Common details')
            # Use Streamlit columns to ensure side-by-side layout inside the form
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Personal details</div>', unsafe_allow_html=True)
                full_name = st.text_input('Full name', key='ef_full_name')
                dob = st.date_input('Date of birth', key='ef_dob', min_value=date(1900,1,1), max_value=date(2100,12,31))
                nationality = st.selectbox('Nationality', country_options, index=0, key='ef_nationality')
                if nationality == 'Other':
                    nationality = st.text_input('Please specify nationality', key='ef_nationality_other')
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Travel & purpose</div>', unsafe_allow_html=True)
                passport_issuing_country = st.selectbox('Passport issuing country', country_options, index=0, key='ef_passport_issuing')
                if passport_issuing_country == 'Other':
                    passport_issuing_country = st.text_input('Please specify issuing country', key='ef_passport_issuing_other')
                passport_issue_date = st.date_input('Passport issue date', key='ef_passport_issue', min_value=date(1900,1,1), max_value=date(2100,12,31))
                passport_expiry_date = st.date_input('Passport expiry date', key='ef_passport_expiry', min_value=date(1900,1,1), max_value=date(2100,12,31))
                current_location = st.selectbox('Current location', ['Inside the UK', 'Outside the UK'], index=1, key='ef_current_location')
                purpose_of_visit = st.selectbox('Purpose of visit / stay', PURPOSE_OPTIONS, key='ef_purpose')
                purpose = purpose_of_visit
                purpose_other = ''
                if purpose_of_visit == 'Other (specify)':
                    purpose_other = st.text_input('Please specify purpose', key='ef_purpose_other')
                    purpose = purpose_other or purpose_of_visit
                travel_start = st.date_input('Planned travel start', key='ef_travel_start', min_value=date(1900,1,1), max_value=date(2100,12,31))
                travel_end = st.date_input('Planned travel end', key='ef_travel_end', min_value=date(1900,1,1), max_value=date(2100,12,31))
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Contact & funds</div>', unsafe_allow_html=True)
                funds_available = st.number_input('Funds available (GBP)', min_value=0.0, value=0.0, key='ef_funds')
                english_proficiency = st.selectbox('Do you meet the required English proficiency?', ['Yes', 'No'], index=0, key='ef_english')
                criminal_convictions = st.selectbox('Any criminal convictions?', ['No', 'Yes'], index=0, key='ef_criminal')
                past_refusals = st.selectbox('Any visa refusals in last 10 years?', ['No', 'Yes'], index=0, key='ef_refusals')
                st.markdown('#### Contact')
                email = st.text_input('Email', key='ef_email')
                phone = st.text_input('Phone', key='ef_phone')
                address = st.text_area('Address', key='ef_address')
                st.markdown('</div>', unsafe_allow_html=True)

            continue_common = st.form_submit_button('Continue')

        # When common details are submitted, persist them in session state and show visa-type selector below
        if continue_common:
            st.session_state.elig_final_common = {
                'full_name': full_name,
                'date_of_birth': dob,
                'nationality': nationality,
                'passport_issuing_country': passport_issuing_country,
                'passport_issue_date': passport_issue_date,
                'passport_expiry_date': passport_expiry_date,
                'current_location': current_location,
                'purpose': purpose,
                'travel_start': travel_start,
                'travel_end': travel_end,
                'funds_available': funds_available,
                'english_proficiency': english_proficiency,
                'criminal_convictions': criminal_convictions,
                'past_refusals': past_refusals,
                'contact': {'email': email, 'phone': phone, 'address': address}
            }
            st.session_state.elig_final_common_submitted = True

        # Show visa-type selector only after common details are submitted
        if st.session_state.get('elig_final_common_submitted'):
            visa_type = st.selectbox('Which visa are you applying for?', ['Student', 'Graduate', 'Skilled Worker', 'Health and Care Worker', 'Standard Visitor'], index=0, key='ef_visa_type_choice')

            # Per-visa sections: only show the fields for the visa selected above
            st.markdown('---')
            
            with st.form(key=f'elig_final_visa_form_{visa_type}'):
                # Graduate
                grad_completed_in_uk = None
                grad_completion_date = None
                grad_current_work = None
                grad_job_title = None

                # Student
                cas_number = ''
                course_start = None
                course_end = None
                sponsor_licensed = 'No'
                tuition_paid = 'No'

                # Skilled Worker
                sw_job_offer = None
                sw_job_title = ''
                sw_soc_code = ''
                sw_salary = 0.0
                sw_employer = ''
                sw_start_date = None

                # Health & Care
                hc_job_offer = None
                hc_registration = ''
                hc_job_title = ''
                hc_soc_code = ''
                hc_salary = 0.0
                hc_employer = ''

                # Visitor
                vis_purpose = ''
                vis_accommodation = 'No'
                vis_return_ticket = 'No'
                vis_length_days = 0

                # Build job title options from JOB_TITLE_TO_SOC and HEALTHCARE_SOC_CODES
                try:
                    sw_titles = sorted(list(JOB_TITLE_TO_SOC.keys()))
                except Exception:
                    sw_titles = []
                try:
                    hc_titles = sorted(list(HEALTHCARE_SOC_CODES.keys()))
                except Exception:
                    hc_titles = []
                # ---- Initialize all visa-specific variables to safe defaults ----

                # Student
                has_cas = None
                cas_reference_number = None
                education_provider_is_licensed = None
                course_level = None
                course_full_time = None
                course_start_date = None
                course_end_date = None
                course_duration_months = None
                meets_financial_requirement = None
                funds_held_for_28_days = None
                english_requirement_met = None

                # Graduate
                currently_in_uk = None
                current_uk_visa_type = None
                course_completed = None
                course_level_completed = None
                provider_reported_completion_to_home_office = None
                original_cas_reference = None
                student_visa_valid_on_application_date = None

                # Skilled Worker
                job_offer_confirmed = None
                employer_is_licensed_sponsor = None
                certificate_of_sponsorship_issued = None
                cos_reference_number = None
                job_title = None
                soc_code = None
                job_is_eligible_occupation = None
                salary_offered = None
                meets_minimum_salary_threshold = None
                criminal_record_certificate_required = None
                criminal_record_certificate_provided = None

                # Health & Care
                employer_is_licensed_healthcare_sponsor = None
                job_is_eligible_healthcare_role = None
                meets_healthcare_salary_rules = None
                professional_registration_required = None
                professional_registration_provided = None

                # Visitor
                purpose_of_visit = None
                purpose_is_permitted_under_visitor_rules = None
                intended_length_of_stay= None
                stay_within_6_months_limit = None
                accommodation_arranged = None
                return_or_onward_travel_planned = None
                intends_to_leave_uk_after_visit = None
                sufficient_funds_for_stay = None

                if visa_type == 'Graduate':
                    st.markdown('#### Graduate')
                    col_a, col_b = st.columns(2)
                    with col_a:
                        currently_in_uk = st.selectbox('Currently in the UK?', ['No','Yes'])
                        current_uk_visa_type = st.selectbox('Current UK visa type',['Student','Tier 4'])
                        course_completed = st.selectbox('Course completed?', ['No','Yes'])
                        course_level_completed = st.selectbox('Course level completed', ['RQF3','RQF4','RQF5','RQF6','RQF7','RQF8'])
                    with col_b:
                        education_provider_is_licensed = st.selectbox('Education provider licensed?', ['No','Yes'])
                        provider_reported_completion_to_home_office = st.selectbox('Provider reported completion?', ['No','Yes'])
                        original_cas_reference = st.text_input('Original CAS reference')
                        student_visa_valid_on_application_date = st.selectbox('Student visa valid on application date?', ['No','Yes'])


                elif visa_type == 'Student':
                    st.markdown('#### Student')
                    col_a, col_b = st.columns(2)
                    with col_a:
                        # Specified student fields as a compact yes/no form (left column)
                        has_cas = st.selectbox('Do you have a CAS?', ['No', 'Yes'], index=0, key='ef_student_has_cas')
                        cas_reference_number = ''
                        if has_cas == 'Yes':
                            cas_reference_number = st.text_input('CAS reference number', key='ef_student_cas_ref')

                        education_provider_is_licensed = st.selectbox('Is the education provider licensed?', ['No', 'Yes'], index=0, key='ef_student_provider_licensed')

                        # Use RQF dropdown for course level per requirements
                        course_level_options = ['RQF3', 'RQF4', 'RQF5', 'RQF6', 'RQF7', 'RQF8']
                        course_level = st.selectbox('Course level (RQF)', course_level_options, index=0, key='ef_student_course_level')

                        course_full_time = st.selectbox('Is the course full-time?', ['No', 'Yes'], index=0, key='ef_student_course_full_time')
                        course_start = st.date_input('Course start date', key='ef_student_course_start', min_value=date(1900,1,1), max_value=date(2100,12,31))
                        course_end = st.date_input('Course end date', key='ef_student_course_end', min_value=date(1900,1,1), max_value=date(2100,12,31))
                    with col_b:
                        course_duration_months = st.number_input('Course duration (months)', min_value=0, value=0, key='ef_student_course_duration_months')

                        meets_financial_requirement = st.selectbox('Do you meet the financial requirement?', ['No', 'Yes'], index=0, key='ef_student_meets_financial')
                        funds_held_for_28_days = st.selectbox('Have the required funds been held for 28 days?', ['No', 'Yes'], index=0, key='ef_student_funds_28')
                        english_requirement_met = st.selectbox('Is the English requirement met?', ['No', 'Yes'], index=0, key='ef_student_english')

                elif visa_type == 'Skilled Worker':
                    st.markdown('#### Skilled Worker')
                    col_a, col_b = st.columns(2)
                    with col_a:
                        job_offer_confirmed = st.selectbox('Job offer confirmed?', ['No', 'Yes'])
                        employer_is_licensed_sponsor = st.selectbox('Employer is licensed sponsor?', ['No', 'Yes'])
                        certificate_of_sponsorship_issued = st.selectbox('Certificate of Sponsorship issued?', ['No', 'Yes'])
                        cos_reference_number = st.text_input('CoS reference number')

                        # CSV-driven dropdown
                        job_title = st.selectbox('Job title', [''] + sorted(JOB_TITLE_TO_SOC.keys()))

                        if 'soc_code' not in st.session_state:
                            st.session_state.soc_code = ''

                        if job_title and st.session_state.get('last_job_title') != job_title:
                            st.session_state.soc_code = JOB_TITLE_TO_SOC.get(job_title, '')
                            st.session_state.last_job_title = job_title

                        soc_code = st.text_input('SOC code(updates automatically according to title no need to fill)', value=st.session_state.soc_code)

                    with col_b:
                        job_is_eligible_occupation = st.selectbox('Job is eligible occupation?', ['No', 'Yes'])
                        salary_offered = st.number_input('Salary offered (£)', min_value=0.0, value=0.0)
                        meets_minimum_salary_threshold = st.selectbox('Meets minimum salary threshold?', ['No', 'Yes'])
                        english_requirement_met = st.selectbox('English requirement met?', ['No', 'Yes'])
                        criminal_record_certificate_required = st.selectbox('Criminal record certificate required?', ['No', 'Yes'])
                        criminal_record_certificate_provided = st.selectbox('Criminal record certificate provided?', ['No', 'Yes'])


                elif visa_type == 'Health and Care Worker':
                    st.markdown('#### Health and Care Worker')
                    col_a, col_b = st.columns(2)
                    with col_a:
                        job_offer_confirmed = st.selectbox('Job offer confirmed?', ['No','Yes'])
                        employer_is_licensed_healthcare_sponsor = st.selectbox('Employer is licensed healthcare sponsor?', ['No','Yes'])
                        certificate_of_sponsorship_issued = st.selectbox('Certificate of Sponsorship issued?', ['No','Yes'])
                        cos_reference_number = st.text_input('CoS reference number')
                        hc_titles = sorted(list(DEFAULT_JOB_TITLE_TO_SOC.keys()))
                        job_title = st.selectbox(
                            "Job title (choose)",
                            ["-- select --"] + hc_titles,
                            key="hc_job_title"
                        )
                        # Auto-fill SOC code from mapping
                        if hc_job_title != "-- select --":
                            st.session_state.hc_soc_code = DEFAULT_JOB_TITLE_TO_SOC.get(job_title, "")
                            st.session_state.hc_last_job_title = hc_job_title
                        else:
                            st.session_state.hc_soc_code = ""
                        # SOC code input (editable)
                        soc_code = st.text_input(
                            "SOC code (auto-filled from job title)",
                            key="hc_soc_code"
                        )

                    with col_b:
                        job_is_eligible_healthcare_role = st.selectbox('Job is eligible healthcare role?', ['No','Yes'])
                        salary_offered = st.number_input('Salary offered', min_value=0.0)
                        meets_healthcare_salary_rules = st.selectbox('Meets healthcare salary rules?', ['No','Yes'])
                        professional_registration_required = st.selectbox('Professional registration required?', ['No','Yes'])
                        professional_registration_provided = st.selectbox('Professional registration provided?', ['No','Yes'])
                        english_requirement_met = st.selectbox('English requirement met?', ['No','Yes'])


                else:  # Standard Visitor
                    st.markdown('#### Visitor')
                    col_a, col_b = st.columns(2)
                    with col_a:
                        purpose_of_visit = st.selectbox(
                            'Purpose of visit / stay',
                            PURPOSE_OPTIONS,
                            key='ef_purpose_f'
                        )
                        purpose_is_permitted_under_visitor_rules = st.selectbox('Purpose permitted under visitor rules?', ['No','Yes'])
                        intended_length_of_stay = st.number_input('Length of stay (days)', min_value=0)

                    with col_b:
                        stay_within_6_months_limit = st.selectbox('Stay within 6 month limit?', ['No','Yes'])
                        accommodation_arranged = st.selectbox('Accommodation arranged?', ['No','Yes'])
                        return_or_onward_travel_planned = st.selectbox('Return or onward travel planned?', ['No','Yes'])
                        intends_to_leave_uk_after_visit = st.selectbox('Intends to leave UK after visit?', ['No','Yes'])
                        sufficient_funds_for_stay = st.selectbox('Sufficient funds for stay?', ['No','Yes'])


                submit_visa = st.form_submit_button('Run eligibility-final check', key=f'btn_run_elig_final_{visa_type}')

            if st.session_state.get('elig_final_common_submitted') and ('submit_visa' in locals() and submit_visa):
                # prepare a compact data dict by merging common fields with visa-specific inputs
                common = st.session_state.get('elig_final_common', {})
                data = {
                        "common": {
                            "english_requirement_met": common.get("english_proficiency") == "Yes",
                            "criminal_history": common.get("criminal_convictions") == "Yes",
                            "previous_refusal": common.get("past_refusals") == "Yes",
                            "funds_available": common.get("funds_available", 0)
                        },

                        "graduate": {
                            "currently_in_uk": currently_in_uk == "Yes",
                            "current_uk_visa_type": current_uk_visa_type,
                            "course_completed": course_completed == "Yes",
                            "course_level_completed": course_level_completed,
                            "education_provider_is_licensed": education_provider_is_licensed == "Yes",
                            "provider_reported_completion_to_home_office": provider_reported_completion_to_home_office == "Yes",
                            "original_cas_reference": original_cas_reference,
                            "student_visa_valid_on_application_date": student_visa_valid_on_application_date == "Yes"
                        },

                        "student": {
                            "has_cas": has_cas == "Yes",
                            "cas_reference_number": cas_reference_number,
                            "education_provider_is_licensed": education_provider_is_licensed == "Yes",
                            "course_level": course_level,
                            "course_full_time": course_full_time == "Yes",
                            "course_start_date": course_start_date,
                            "course_end_date": course_end_date,
                            "course_duration_months": course_duration_months,
                            "meets_financial_requirement": meets_financial_requirement == "Yes",
                            "funds_held_for_28_days": funds_held_for_28_days == "Yes",
                            "english_requirement_met": english_requirement_met == "Yes"
                        },

                        "skilled_worker": {
                            "job_offer_confirmed": job_offer_confirmed == "Yes",
                            "employer_is_licensed_sponsor": employer_is_licensed_sponsor == "Yes",
                            "certificate_of_sponsorship_issued": certificate_of_sponsorship_issued == "Yes",
                            "cos_reference_number": cos_reference_number,
                            "job_title": job_title,
                            "soc_code": soc_code,
                            "job_is_eligible_occupation": job_is_eligible_occupation == "Yes",
                            "salary_offered": salary_offered,
                            "meets_minimum_salary_threshold": meets_minimum_salary_threshold == "Yes",
                            "english_requirement_met": english_requirement_met == "Yes",
                            "criminal_record_certificate_required": criminal_record_certificate_required == "Yes",
                            "criminal_record_certificate_provided": criminal_record_certificate_provided == "Yes"
                        },

                        "health_care": {
                            "job_offer_confirmed": job_offer_confirmed == "Yes",
                            "employer_is_licensed_healthcare_sponsor": employer_is_licensed_healthcare_sponsor == "Yes",
                            "certificate_of_sponsorship_issued": certificate_of_sponsorship_issued == "Yes",
                            "cos_reference_number": cos_reference_number,
                            "job_title": job_title,
                            "soc_code": soc_code,
                            "job_is_eligible_healthcare_role": job_is_eligible_healthcare_role == "Yes",
                            "salary_offered": salary_offered,
                            "meets_healthcare_salary_rules": meets_healthcare_salary_rules == "Yes",
                            "professional_registration_required": professional_registration_required == "Yes",
                            "professional_registration_provided": professional_registration_provided == "Yes",
                            "english_requirement_met": english_requirement_met == "Yes"
                        },

                        "visitor": {
                            "purpose_of_visit": purpose_of_visit,
                            "purpose_is_permitted_under_visitor_rules": purpose_is_permitted_under_visitor_rules == "Yes",
                            "intended_length_of_stay": intended_length_of_stay,
                            "stay_within_6_months_limit": stay_within_6_months_limit == "Yes",
                            "accommodation_arranged": accommodation_arranged == "Yes",
                            "return_or_onward_travel_planned": return_or_onward_travel_planned == "Yes",
                            "intends_to_leave_uk_after_visit": intends_to_leave_uk_after_visit == "Yes",
                            "sufficient_funds_for_stay": sufficient_funds_for_stay == "Yes"
                        }
                    }


                st.session_state.elig_final_form = data

                # Choose and call the corresponding evaluator
                try:
                    if visa_type == 'Student':
                        result = evaluate_student('core', data)
                    elif visa_type == 'Graduate':
                        result = evaluate_graduate('core', data)
                    elif visa_type == 'Skilled Worker':
                        result = evaluate_skilled_worker('core', data)
                    elif visa_type == 'Health and Care Worker':
                        result = evaluate_health_care('core', data)
                    else:
                        # Standard Visitor
                        result = evaluate_visitor('core', data)
                except Exception as e:
                    st.error('An error occurred while evaluating eligibility. See details below.')
                    st.exception(e)
                    result = {'eligible': False, 'passed_rules': [], 'failed_rules': ['EVALUATION_ERROR']}

                st.session_state.elig_final_result = result

                # If failed rules, retrieve supporting policy chunks and get LLM explanation
                if result.get('failed_rules'):
                    st.error('❌ Eligibility checks failed')
                    failed = result.get('failed_rules', [])
                    # quick short reasons (best-effort)
                    RULE_SHORT = {
                        'FUNDS_INSUFFICIENT': 'insufficient funds',
                        'VALID_PASSPORT': 'passport is invalid or expired',
                        'INTENDS_TO_LEAVE': 'no clear intention to leave the UK',
                        'RETURN_TRAVEL': 'no return travel booked',
                        'EVIDENCE_MISSING': 'required evidence not provided'
                    }
                    reasons = [RULE_SHORT.get(r, f'failed requirement {r}') for r in failed]
                    if reasons:
                        st.markdown('**Quick reason:** ' + '; '.join(reasons))

                    # Retrieve policy chunks (fallback to services.retrieval)
                    try:
                        # Prefer the RAG-aware retrieval helper if available (returns rich chunk metadata)
                        # Pass the visa_type so RAG can prioritise matched visa chunks.
                        if 'retrieve_with_rag' in globals():
                            retrieved = retrieve_with_rag(failed, visa_type=visa_type, top_k=3)
                        else:
                            retrieved = retrieve_policy_chunks(failed, visa_type=visa_type, top_k=3)
                    except Exception:
                        retrieved = []
                    st.session_state.elig_final_retrieved = retrieved

                    # Ask LLM for a concise explanation (on-demand style: call here but it's fast in our services)
                    try:
                        expl = llm_explain({'rule_results': result}, retrieved)
                    except Exception:
                        expl = None
                    st.session_state.elig_final_explanation = expl

                else:
                    st.success('✅ You meet the configured eligibility checks for this visa type')
                    st.session_state.elig_final_retrieved = []
                    st.session_state.elig_final_explanation = None

        # Render results if present
        if st.session_state.get('elig_final_result'):
            r = st.session_state.elig_final_result
            # st.markdown('---')
            # st.markdown('### Result')
            # st.write(r)
            if st.session_state.get('elig_final_explanation'):
                expl = st.session_state.elig_final_explanation
                st.markdown('### LLM Explanation')
                st.markdown(f"**Decision:** {expl.get('decision')}")
                # Prefer per-rule explanations and recommendations only
                if expl.get('per_rule'):
                    # st.markdown('#### explanations')
                    for pr in expl.get('per_rule'):
                        st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                        cit = pr.get('citation') or {}
                        if cit:
                            st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                            st.markdown(f"  - Page: {cit.get('page','N/A')}")
                            if cit.get('section'):
                                st.markdown(f"  - Section: {cit.get('section')}")
                if expl.get('recommendations'):
                    st.markdown('#### Recommendations')
                    for rec in expl.get('recommendations'):
                        st.markdown(f"- {rec}")
                if not expl.get('per_rule') and not expl.get('recommendations'):
                    st.markdown(f"**Summary:** {expl.get('summary')}")
                if st.session_state.get('elig_final_retrieved'):
                    st.markdown('#### Supporting citations')
                    # Show a concise snippet + metadata for each retrieved chunk (like Query tab)
                    for c in st.session_state.elig_final_retrieved:
                        doc = c.get('doc', 'Unknown')
                        page = c.get('page', 'N/A')
                        section = c.get('section', '')
                        text = c.get('text') or c.get('content') or ''
                        with st.container():
                            st.markdown(f"**{doc}** (page: {page})")
                            if section:
                                st.caption(section)
                            if text:
                                # show a short excerpt
                                excerpt = text if len(text) < 400 else text[:400] + '...'
                                st.markdown(f"_{excerpt}_")
                            else:
                                st.caption('No extracted paragraph available for this citation')

                    # Add an expander to show the raw retrieved objects for debugging
                    with st.expander('Show raw retrieved chunks (debug)'):
                        try:
                            st.write(st.session_state.elig_final_retrieved)
                        except Exception:
                            st.text(str(st.session_state.elig_final_retrieved))
                else:
                    # Do NOT re-evaluate inside render. The final evaluation should already
                    # have been run when the user submitted the per-visa form. Here we only
                    # render the stored `elig_final_*` session state and offer an on-demand
                    # LLM explanation button when there are failed rules and no explanation yet.
                    result = st.session_state.get('elig_final_result') or {}

                    # Show Rule results summary
                    st.markdown('---')
                    st.markdown('### Rule results')
                    for rr in result.get('passed_rules', []):
                        st.markdown(f"- ✅ {rr}")
                    for rr in result.get('failed_rules', []):
                        st.markdown(f"- ❌ {rr}")

                    # If there are failed rules, allow the user to request a detailed LLM explanation
                    if result.get('failed_rules'):
                        # Show a request button if no explanation exists yet
                        if not st.session_state.get('elig_final_explanation'):
                            if st.button('Request detailed LLM explanation', key=f'btn_request_llm_final_{visa_type}'):
                                with st.spinner('Generating LLM explanation...'):
                                    retrieved = retrieve_with_rag(result.get('failed_rules', []), visa_type=visa_type)
                                    st.session_state.elig_final_retrieved = retrieved
                                    expl = llm_explain({'rule_results': result}, retrieved)
                                    st.session_state.elig_final_explanation = expl

                        # Render explanation if available
                        if st.session_state.get('elig_final_explanation'):
                            expl = st.session_state.elig_final_explanation
                            st.markdown('### Explanation')
                            st.markdown(f"**Decision:** {expl.get('decision')}")
                            # Show only per-rule explanations and recommendations (no raw JSON)
                            if expl.get('per_rule'):
                                st.markdown('#### Per-rule explanations')
                                for pr in expl.get('per_rule'):
                                    st.markdown(f"- **{pr.get('rule')}**: {pr.get('explanation')}")
                                    cit = pr.get('citation') or {}
                                    if cit:
                                        st.markdown(f"  - Document: {cit.get('doc','N/A')}")
                                        st.markdown(f"  - Page: {cit.get('page','N/A')}")
                                        if cit.get('section'):
                                            st.markdown(f"  - Section: {cit.get('section')}")
                            if expl.get('recommendations'):
                                st.markdown('#### Recommendations')
                                for rec in expl.get('recommendations'):
                                    st.markdown(f"- {rec}")
                            if not expl.get('per_rule') and not expl.get('recommendations'):
                                st.markdown(f"**Summary:** {expl.get('summary')}")

                    # Supporting citations (if any) for eligibility-final
                    if st.session_state.get('elig_final_retrieved'):
                        st.markdown('### Supporting citations')
                        for c in st.session_state.elig_final_retrieved:
                            st.markdown('**Document:** ' + str(c.get('doc', 'Unknown')))
                    # Detailed checks form removed per final-tab UX: we only keep the single
                    # per-visa form that is submitted from the eligibility-final tab. This
                    # duplicate detailed form introduced confusion and is not required.
                if st.button('Back to Core', key='btn_back_to_core_from_detailed'):
                    st.session_state.elig_step = 'core'



if __name__ == "__main__":
    main()
