# --------------------------------------------------- End-to-End OpenAI Multi-modal RAG App ---------------------------------------------------
import streamlit as st
import json
import os
import time
import faiss
import fitz
import warnings
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults

# ---------------------------------------------------------- Load Keys ----------------------------------------------------------
def load_key_from_file(file_name: str, env_var: str):
    try:
        with open(file_name, "r") as f:
            key = f.read().strip()
            os.environ[env_var] = key
    except Exception:
        pass

for file, env in [
    ("OPENAI_API_KEY.txt", "OPENAI_API_KEY"),
    ("TAVILY_API_KEY.txt", "TAVILY_API_KEY"),
]:
    load_key_from_file(file, env)

load_dotenv()
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------------------------------- Load and Chunk PDFs ----------------------------------------------------------
# Instatiate the loader
loader = PyMuPDFLoader('/Users/emmanueldanielchonza/Documents/GenAI-AV-capstone-project/data/attention_paper.pdf')
docs = loader.load()
# docs[:1]

pdfs = []
for root, dirs, files in os.walk('data'):
    for file in files:
        if file.endswith('.pdf'):
            pdfs.append(os.path.join(root, file))

docs = []
for path in pdfs:
    with fitz.open(path) as d:
        pages = [p.get_text("text") for p in d]
        docs.append({'file_name': path, 'pages': pages})

# Initialize embeddings (OpenAI)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------- Chunking PDFs ----------------------------------------------------------
st.info("Loading and chunking local PDF documents...")
pdfs = []
for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))

docs = []
for file_path in pdfs:
    with fitz.open(file_path) as document:
        pages_content = [page.get_text("text") for page in document]
        docs.append({"file_name": file_path, "pages": pages_content})

# Create text chunks for embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = []
for doc in docs:
    for content in doc["pages"]:
        chunked_texts = splitter.split_text(content)
        for i, chunk in enumerate(chunked_texts):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": doc["file_name"],
                        "page": i + 1,
                        "chunk_length": len(chunk),
                    },
                )
            )

# -------------------------------------------------- Create or Load Chroma Vector Store --------------------------------------------------
# Directory to persist the vector database
CHROMA_DB_DIR = "chromaDB"

# Check if the vector database already exists
if os.path.exists(CHROMA_DB_DIR):
    st.success("Found existing ChromaDB index!!...")
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
else:
    st.warning("⚙️ No existing ChromaDB found!!...")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()
    st.success(f"Vector database created and persisted at ChromaDB")
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 100, "lambda_mult": 0.6})

# --------------------------------------------- Prompt ----------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant for question-answering tasks.
Use ONLY the following context (retrieved, image, and table) to answer the question.
Cite properly the local file or web-link the answer is obtained from.
Avoid hallucinating. If unsure, say "I do not know."

Format:
- Bullet points, concise and professional.
- Cite sources inline using (page N, FILE.pdf) for local and (URL, paper title) for web.

Question: {question}

Context:
{context}

Image Info:
{image_context}

Table Info:
{table_context}

Answer:""")

def format_docs(docs):
    lines = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source_file", doc.metadata.get("source", ""))
        lines.append(f"[{i}] (p{page}, {source}) {doc.page_content}")
    return "\n\n".join(lines)

def handle_image_upload(path): return f"OCR or caption extracted from: {path}" if path else ""
def handle_table_upload(rows): return "\n".join(", ".join(f"{k}: {v}" for k, v in zip(r["columns"], r["values"])) for r in rows) if rows else "No table data."

def fallback_web_augmented_context(query: str, min_k=2):
    local_docs = retriever.invoke(query)
    if len(local_docs) >= min_k:
        return format_docs(local_docs)

    web_tool = TavilySearchResults() if os.getenv("TAVILY_API_KEY") else DuckDuckGoSearchResults()
    results = web_tool.invoke({"query": query})
    formatted_web = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        formatted_web.append(f"[{i}] ({title}) {content} ({url})")
    return "\n\n".join(formatted_web)

# llm_gpt = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
llm_gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": RunnableLambda(lambda x: fallback_web_augmented_context(x["question"])),
        "image_context": RunnableLambda(lambda x: handle_image_upload(x.get("image_path", ""))),
        "table_context": RunnableLambda(lambda x: handle_table_upload(x.get("table_data", []))),
    }
    | prompt
    | llm_gpt
    | StrOutputParser()
)

# ---------------------------------------------------------- Streamlit UI ----------------------------------------------------------
st.set_page_config(page_title="OpenAI Multi-Modal RAG", layout="wide")
st.markdown("<h2 style='text-align:center;'>🔎 OpenAI Multi-Modal RAG</h2>", unsafe_allow_html=True)

if "chat" not in st.session_state:
    st.session_state.chat = {}
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

with st.sidebar:
    st.header("📥 Data Inputs")
    img = st.file_uploader("📷 Upload Image", type=["png", "jpg", "jpeg"])
    pdf = st.file_uploader("📄 Upload .pdf or .docx", type=["pdf", "docx"])
    table_json = st.text_area("📊 Paste table data (JSON)", height=100)
    min_k = st.slider("🔎 Min Local Docs before Web Search", 0, 5, 2)
    session = st.text_input("🗂️ Session Name", "openai-session")

session_chat = st.session_state.chat.setdefault(session, [])

st.subheader("💬 Ask Your Question")
with st.form("ask_form", clear_on_submit=True):
    query = st.text_area("Enter your question here:", height=100, key="query_input")
    submitted = st.form_submit_button("🚀 Run Query")

if submitted and query.strip():
    table_data = []
    try:
        if table_json:
            table_data = json.loads(table_json)
    except json.JSONDecodeError:
        st.error("❌ Invalid table JSON")

    image_path = None
    if img:
        image_path = f".cache/img_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(img.read())

    if pdf:
        st.success(f"📄 {pdf.name} uploaded. (Indexing not yet handled)")

    payload = {
        "question": query,
        "image_path": image_path or "",
        "table_data": table_data,
    }

    try:
        with st.spinner("🤖 Generating answer..."):
            result = rag_chain.invoke(payload)
            context = fallback_web_augmented_context(query, min_k=min_k)
        session_chat.append({"ts": time.time(), "query": query, "response": result, "context": context})
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.subheader("Chat History")
for turn in reversed(session_chat[-10:]):
    st.markdown(f"**Q:** {turn['query']}")
    st.markdown(f"**A:** {turn['response']}")
    with st.expander("📎 Retrieved Context"):
        st.code(turn['context'], language="markdown")

st.sidebar.markdown("---")
if session_chat:
    export = json.dumps(session_chat, indent=2)
    st.sidebar.download_button("💾 Download Chat", export, file_name=f"chat_{session}.json", mime="application/json")
