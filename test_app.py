import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Streamlit Page Config
st.set_page_config(page_title="PDF RAG with Gemini", layout="wide")

st.title("📄 PDF QA Assistant (Powered by LangChain & Gemini)")
st.write("Upload a PDF document, ask questions, and get precise answers along with source page numbers.")

# ==========================================
# ENTER YOUR ACTUAL GEMINI API KEY HERE:
# ==========================================
# Replace DIRECT_API_KEY with st.secrets
try:
  google_api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
  st.error(
      "🚨 GOOGLE_API_KEY is missing from Streamlit Secrets. Please add it in"
      " your app settings."
  )
  st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Update to a valid model name
    google_api_key=google_api_key,
    temperature=0,
)

# Sidebar for File Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

@st.cache_resource
def initialize_vector_store(file_path):
    """Loads PDF, splits chunks, and creates a Chroma vector store."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store

if uploaded_file is not None:
    # Save uploaded file temporarily to process with PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing PDF and building vector index... Please wait."):
        vector_store = initialize_vector_store(tmp_file_path)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=google_api_key,
        temperature=0
    )

    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.success("PDF successfully processed! Ask your questions below.")

    # User Query Input
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):
            # Get retrieved documents separately to extract page numbers
            retrieved_docs = retriever.invoke(query)
            
            # Generate the response using your RAG chain
            response = rag_chain.invoke(query)

        # Display Answer
        st.subheader("Answer:")
        st.write(response)

        # Display Source Pages & Context
        st.subheader("Sources & Page Numbers:")
        for i, doc in enumerate(retrieved_docs):
            page_num = doc.metadata.get("page", 0) + 1  # Adding 1 for user-friendly 1-indexed pages
            with st.expander(f"Source {i+1} (Page {page_num})"):
                st.write(doc.page_content)
else:
    st.info("Please upload a PDF file in the sidebar to get started.")
