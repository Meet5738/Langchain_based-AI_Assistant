import streamlit as st
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI Cyber Assistant", layout="wide")

st.title("AI Cybersecurity Assistant")

groq_key = st.text_input("Enter Groq API Key", type="password")

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

# Upload files
uploaded_files = st.file_uploader(
    "Upload Text Files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files and groq_key:
    documents = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.read())

        loader = TextLoader(file.name, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file.name

        documents.extend(docs)

    st.success(f"{len(uploaded_files)} files uploaded!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    query = st.text_input("Ask your question:")

    if query:
        relevant_docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        Answer based only on the context.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Sources")

        for i, doc in enumerate(relevant_docs):
            st.markdown(f"**Source {i+1}: {doc.metadata['source']}**")
            st.write(doc.page_content[:300] + "...")
            st.divider()
