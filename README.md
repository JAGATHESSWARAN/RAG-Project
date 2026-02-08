# RAG Chatbot Project

A simple Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Streamlit. Users upload files (PDF, Word, Excel, CSV), and the bot answers questions about the content naturally.

## Prerequisites
- Python 3.10+
- Ollama (install from ollama.ai, pull model with `ollama pull llama2`)
- Install dependencies: `pip install -r requirements.txt`

## How to Run
1. Start Ollama: `ollama serve`
2. Run the app: `streamlit run app.py`
3. Open in browser (localhost:8501)
4. Upload a file and chat!

## Folder Structure
- app.py: Main script
- requirements.txt: Dependencies
- uploads/: For temporary file storage (created at runtime)