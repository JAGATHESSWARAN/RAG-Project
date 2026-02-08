import streamlit as st
import os

# ---------- LangChain imports ----------
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- App setup ----------
st.set_page_config(page_title="RAG Chatbot (Groq)", layout="wide")
st.title("📄 RAG Chatbot")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- API KEY CHECK ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY not found.\n\n"
        "Please set the GROQ_API_KEY environment variable and restart the app."
    )
    st.stop()

# ---------- File loader ----------
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    elif ext in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")

# ---------- Prompt ----------
PROMPT_TEMPLATE = """
You are a helpful AI assistant.
Answer the question strictly using the given context.
If the answer is not present, say:
"I don't have that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

# ---------- Session state ----------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- File upload ----------
uploaded_file = st.file_uploader(
    "Upload a file in PDF Format",
    type=["pdf", "docx", "csv", "xlsx", "xls"]
)

# ---------- Build RAG ----------
if uploaded_file and st.session_state.rag_chain is None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔄 Processing document..."):
        documents = load_document(file_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0
)



        st.session_state.rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    st.success("✅ Document processed. You can start chatting!")

# ---------- Chat UI ----------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.rag_chain:
    user_query = st.chat_input("Ask a question about the document...")

    if user_query:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag_chain.invoke(user_query)
                st.markdown(response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
