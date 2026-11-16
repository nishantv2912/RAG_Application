import hashlib
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Chat with your document", page_icon="📄", layout="wide")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, runs locally on CPU

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the provided context excerpts from the "
            "user's document. Cite the excerpts you used like [1], [2]. If the answer "
            "is not in the context, say you couldn't find it in the document. "
            "Be concise and accurate.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


# ----------------------------- Loading / indexing -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
    )


def load_documents(uploaded):
    """Load an uploaded file into LangChain Documents using the matching loader."""
    suffix = os.path.splitext(uploaded.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()
    finally:
        os.remove(path)

    for d in docs:
        d.metadata["source"] = uploaded.name  # replace temp path with real name
    return docs


def build_vectorstore(files, chunk_size, chunk_overlap):
    docs = []
    for f in files:
        docs.extend(load_documents(f))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        return None, 0
    store = InMemoryVectorStore.from_documents(chunks, get_embeddings())
    return store, len(chunks)


def label(doc):
    page = doc.metadata.get("page")
    return f"{doc.metadata.get('source', 'file')}" + (f" p.{page + 1}" if page is not None else "")


def format_context(docs):
    return "\n\n".join(f"[{i}] (source: {label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))


# ----------------------------- UI -----------------------------
st.title("📄 Chat with your document")
st.caption("LangChain + Gemini RAG: upload files, ask a question, get a grounded answer.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
        help="Free key from https://aistudio.google.com/apikey. Leave empty to see only retrieved passages.",
    )
    model = st.text_input(
        "Gemini model",
        value="gemini-3.1-flash-lite",
        help="Use any model your free key can access (check AI Studio), e.g. a Flash or Flash-Lite model.",
    )
    top_k = st.slider("Passages to retrieve", 1, 10, 4)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 200, step=50)

uploaded_files = st.file_uploader(
    "Upload documents", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_files:
    signature = hashlib.md5(
        b"".join(f.getvalue() for f in uploaded_files)
        + f"{chunk_size}-{chunk_overlap}".encode()
    ).hexdigest()

    if st.session_state.get("signature") != signature:
        with st.spinner("Reading and indexing your documents..."):
            try:
                store, n = build_vectorstore(uploaded_files, chunk_size, chunk_overlap)
            except Exception as e:
                store, n = None, 0
                st.error(f"Could not process the files: {e}")
        st.session_state.store = store
        st.session_state.signature = signature
        st.session_state.messages = []
        if store is not None:
            st.success(f"Indexed {n} chunks from {len(uploaded_files)} file(s).")
        else:
            st.error("No readable text found. Scanned PDFs need OCR first.")
else:
    st.session_state.pop("store", None)
    st.session_state.pop("signature", None)
    st.info("Upload a PDF, DOCX, TXT or MD file to get started.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input(
    "Ask a question about your document...", disabled=not st.session_state.get("store")
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    store = st.session_state.store
    hits = store.similarity_search_with_score(question, k=top_k)  # [(Document, score)]
    docs = [d for d, _ in hits]

    with st.chat_message("assistant"):
        if api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=model, google_api_key=api_key, temperature=0.2, max_output_tokens=1000
                )
                chain = PROMPT | llm | StrOutputParser()
                answer = st.write_stream(
                    chain.stream({"context": format_context(docs), "question": question})
                )
            except Exception as e:
                answer = f"Error calling the Gemini API: {e}"
                st.error(answer)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.warning("Free-tier rate limit hit. Wait a minute, or try a Flash-Lite model.")
        else:
            answer = "No API key set, so here are the most relevant passages from your document:"
            st.markdown(answer)

        with st.expander("Retrieved passages"):
            for i, (doc, score) in enumerate(hits, 1):
                st.markdown(f"**[{i}] {label(doc)}** · similarity {score:.2f}")
                st.write(doc.page_content)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})