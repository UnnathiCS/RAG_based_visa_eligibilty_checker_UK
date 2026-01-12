#!/bin/bash
# Quick launcher for Streamlit RAG Interface
# Usage: bash run_streamlit.sh

echo " Visa Policy RAG - Streamlit Interface"
echo "========================================"
echo ""

# Check if Ollama is running
echo "🔍 Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama is NOT running"
    echo ""
    echo "Start Ollama in a new terminal:"
    echo "   ollama serve"
    echo ""
    echo "Then pull Mistral (one-time):"
    echo "   ollama pull mistral"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo " Starting Streamlit..."
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run streamlit
streamlit run streamlit_app.py --logger.level=warning
