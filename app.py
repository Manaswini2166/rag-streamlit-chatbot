import streamlit as st
from rag_utils import (
    load_and_split_pdf,
    create_vectorstore,
    create_retriever,
    load_llm,
    create_rag_chain
)

# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

st.title("🤖 RAG Chatbot")

# -----------------------------------
# File Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    # Save uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Process PDF
    with st.spinner("Processing and indexing document..."):
        docs = load_and_split_pdf("temp.pdf")
        vectorstore = create_vectorstore(docs)
        retriever = create_retriever(vectorstore)
        llm = load_llm()
        rag_chain = create_rag_chain(llm, retriever)

    st.success("Document indexed successfully!")

    # -----------------------------------
    # Question Input
    # -----------------------------------
    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Generating answer..."):
            answer = rag_chain.run(query)

        st.subheader("Answer")
        st.write(answer)
