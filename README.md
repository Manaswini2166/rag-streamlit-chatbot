# 📄 RAG-Based PDF Chatbot using Streamlit & Ollama

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload PDF documents and ask context-aware questions based on the document content.

The system uses **FAISS** for semantic search, **sentence-transformer embeddings** for vectorization, and a **local LLM powered by Ollama**, eliminating dependency on external APIs.

---

## 🚀 Features
- Upload and process PDF documents
- Context-aware question answering
- Semantic retrieval using FAISS
- Local LLM inference using Ollama
- Interactive Streamlit web interface

---

## 🛠 Tech Stack
- Python
- LangChain
- Streamlit
- FAISS
- Sentence-Transformers
- Ollama (Local LLM)

---

## ▶️ How to Run

```bash
# Create virtual environment
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull mistral

# Run the app
streamlit run app.py
