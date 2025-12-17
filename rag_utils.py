"""
rag_utils.py
-------------
RAG utility functions using:
- PDF Loader
- Text Splitter
- FAISS Vector Store
- Ollama (Local LLM)
"""

# -----------------------------
# Document Loader
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# Text Splitter
# -----------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Embeddings & Vector Store
# -----------------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Local LLM (OLLAMA)
# -----------------------------
from langchain_community.llms import Ollama

# -----------------------------
# RAG Chain
# -----------------------------
from langchain.chains import RetrievalQA


# ==================================================
# 1. Load PDF and split into chunks
# ==================================================
def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(documents)
    return split_docs


# ==================================================
# 2. Create Vector Store (FAISS)
# ==================================================
def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


# ==================================================
# 3. Create Retriever
# ==================================================
def create_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return retriever


# ==================================================
# 4. Load Local LLM (Ollama)
# ==================================================
def load_llm():
    llm = Ollama(
        model="mistral",   # you can change to phi3, llama3.1, gemma
        temperature=0.4
    )
    return llm


# ==================================================
# 5. Create RAG Chain
# ==================================================
def create_rag_chain(llm, retriever):
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return rag_chain
