#!/bin/bash
# Setup script for Streamlit RAG Interface
# Run with: bash setup_streamlit.sh

echo "🛂 Visa Policy RAG System - Streamlit Setup"
echo "============================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "📥 Installing required packages..."
echo ""

# Install packages
pip install --upgrade pip

packages=(
    "streamlit>=1.28.0"
    "faiss-cpu>=1.7.4"
    "sentence-transformers>=2.2.2"
    "PyPDF2>=3.0.1"
    "requests>=2.31.0"
    "torch>=2.0.0"
    "numpy>=1.24.0"
    "scikit-learn>=1.3.0"
)

for package in "${packages[@]}"; do
    echo "  Installing: $package"
    pip install "$package"
done

echo ""
echo "✅ Installation complete!"
echo ""

# Check databases
echo "🔍 Checking RAG databases..."
echo ""

if [ -d "visa_db" ] && [ -f "visa_db/faiss.index" ]; then
    chunks=$(python3 -c "import pickle; f=open('visa_db/chunks.pkl','rb'); data=pickle.load(f); print(len(data))" 2>/dev/null || echo "?")
    echo "  ✓ India database: $chunks chunks"
else
    echo "  ⚠️  India database not found (will create on first run)"
fi

if [ -d "uk_visa_db" ] && [ -f "uk_visa_db/faiss.index" ]; then
    chunks=$(python3 -c "import pickle; f=open('uk_visa_db/chunks.pkl','rb'); data=pickle.load(f); print(len(data))" 2>/dev/null || echo "?")
    echo "  ✓ UK database: $chunks chunks"
else
    echo "  ⚠️  UK database not found (will create on first run)"
fi

echo ""
echo "🚀 Ready to launch!"
echo ""
echo "Next steps:"
echo "  1. Start Ollama in another terminal: ollama serve"
echo "  2. Pull Mistral model (if needed): ollama pull mistral"
echo "  3. Run the Streamlit app: streamlit run streamlit_app.py"
echo ""
echo "The app will open at: http://localhost:8501"
echo ""
