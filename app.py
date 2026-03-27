import streamlit as st
import urllib.request
import io
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat · RAG Pipeline",
    page_icon="📄",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0f14;
    color: #e8e4dc;
}

/* Header */
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.6rem !important;
    color: #f0ebe0 !important;
    letter-spacing: -0.5px;
    line-height: 1.15 !important;
}

h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 300 !important;
    color: #8b8680 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #13161d !important;
    border-right: 1px solid #1f2330;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #8b8680;
    font-size: 0.85rem;
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #1a1d26 !important;
    border: 1px solid #2a2e3d !important;
    border-radius: 6px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #c4a882 !important;
    box-shadow: 0 0 0 1px #c4a88240 !important;
}

/* Buttons */
.stButton > button {
    background-color: #c4a882 !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 5px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s;
}

.stButton > button:hover {
    background-color: #d4bc9a !important;
    transform: translateY(-1px);
}

/* Answer box */
.answer-box {
    background: #13161d;
    border: 1px solid #2a2e3d;
    border-left: 3px solid #c4a882;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #e8e4dc;
    margin-top: 1rem;
}

/* Context expander */
.streamlit-expanderHeader {
    background-color: #1a1d26 !important;
    color: #8b8680 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Success / info / warning */
.stSuccess, .stInfo {
    background-color: #1a1d26 !important;
    border-color: #2a2e3d !important;
    color: #e8e4dc !important;
}

/* Divider */
hr {
    border-color: #1f2330 !important;
}

/* Step tags */
.step-tag {
    display: inline-block;
    background: #1a1d26;
    border: 1px solid #2a2e3d;
    color: #c4a882;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 6px;
}

/* Chat history */
.chat-user {
    background: #1a1d26;
    border: 1px solid #2a2e3d;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0 0.3rem 0;
    font-size: 0.9rem;
    color: #c4a882;
}

.chat-bot {
    background: #13161d;
    border: 1px solid #2a2e3d;
    border-left: 3px solid #4a7fa5;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    margin: 0.3rem 0 0.8rem 0;
    font-size: 0.9rem;
    color: #e8e4dc;
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)


# ── Pipeline setup (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(pdf_source: str, source_type: str, groq_api_key: str):
    """Load PDF, chunk, embed, build FAISS index. Cached per (source, key)."""
    from pypdf import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    import faiss
    import numpy as np

    # 1 ── Load PDF ─────────────────────────────────────────────────────────────
    if source_type == "url":
        req = urllib.request.Request(pdf_source, headers={"User-Agent": "Mozilla/5.0"})
        pdf_bytes = urllib.request.urlopen(req).read()
    else:  # uploaded file bytes
        pdf_bytes = pdf_source

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages)

    # 2 ── Chunk ────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    # 3 ── Embed ────────────────────────────────────────────────────────────────
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = np.array(embedder.embed_documents(chunks), dtype="float32")

    # 4 ── FAISS index ──────────────────────────────────────────────────────────
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return chunks, index, embedder, len(pages), groq_api_key


def ask_pdf(question: str, chunks, index, embedder, groq_api_key: str, k: int = 5):
    """Retrieve top-k chunks and answer with Groq."""
    import numpy as np
    from groq import Groq

    # Embed the question
    q_vec = np.array([embedder.embed_query(question)], dtype="float32")
    _, indices = index.search(q_vec, k)
    context = "\n\n---\n\n".join([chunks[i] for i in indices[0]])

    # Ask Groq
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document assistant. Answer the question using ONLY "
                    "the provided context. If the answer is not in the context, say so clearly. "
                    "Be concise but complete."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=512,
    )
    answer = response.choices[0].message.content
    return answer, context


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("# 📄 Chat with a PDF")
st.markdown("### RAG pipeline · PDF → Chunks → Embeddings → FAISS → Groq")
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )

    st.markdown("---")
    st.markdown("### 📎 PDF Source")

    source_option = st.radio(
        "Load PDF from:",
        ["URL", "Upload file"],
        horizontal=True,
    )

    pdf_url = ""
    uploaded_file = None

    if source_option == "URL":
        pdf_url = st.text_input(
            "PDF URL",
            value="https://arxiv.org/pdf/1706.03762",
            placeholder="https://...",
        )
    else:
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    top_k = st.slider("Chunks to retrieve (k)", min_value=3, max_value=10, value=5)

    load_btn = st.button("🔄 Load & Index PDF", use_container_width=True)

    st.markdown("---")
    st.markdown("""
**Stack**  
`pypdf` · `LangChain` · `HuggingFace`  
`FAISS` · `Groq llama-3.3-70b`

**Pipeline**  
PDF → Load → Chunk → Embed → Index → Retrieve → Answer
""")

# ── Session state ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page_count" not in st.session_state:
    st.session_state.page_count = 0

# ── Load pipeline ──────────────────────────────────────────────────────────────
if load_btn:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
    elif source_option == "URL" and not pdf_url:
        st.error("⚠️ Please enter a PDF URL.")
    elif source_option == "Upload file" and uploaded_file is None:
        st.error("⚠️ Please upload a PDF file.")
    else:
        with st.spinner("Loading PDF · Chunking · Embedding · Building FAISS index..."):
            try:
                if source_option == "URL":
                    src, stype = pdf_url, "url"
                else:
                    src, stype = uploaded_file.read(), "bytes"

                chunks, index, embedder, page_count, key = load_pipeline(
                    src, stype, groq_api_key
                )
                st.session_state.pipeline = (chunks, index, embedder, key)
                st.session_state.page_count = page_count
                st.session_state.chat_history = []
                st.success(
                    f"✅ Indexed {page_count} pages → {len(chunks)} chunks. Ready to chat!"
                )
            except Exception as e:
                st.error(f"❌ Error loading PDF: {e}")

# ── Chat area ──────────────────────────────────────────────────────────────────
if st.session_state.pipeline:
    chunks, index, embedder, key = st.session_state.pipeline

    # Render chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for turn in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-user">🧑 {turn["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-bot">🤖 {turn["answer"]}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("📑 Retrieved context", expanded=False):
                st.code(turn["context"][:1200] + "…" if len(turn["context"]) > 1200 else turn["context"],
                        language="text")
        st.markdown("---")

    # Question input
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Ask a question about the PDF",
            placeholder="e.g. What is the main contribution of this paper?",
            label_visibility="collapsed",
        )
    with col2:
        ask_btn = st.button("Ask →", use_container_width=True)

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            "What is the main contribution of this paper?",
            "Explain the architecture in simple terms.",
            "What results did they achieve?",
        ]
        cols = st.columns(3)
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"sug_{i}"):
                question = s
                ask_btn = True

    if ask_btn and question:
        with st.spinner("Retrieving · Generating answer…"):
            try:
                answer, context = ask_pdf(question, chunks, index, embedder, key, k=top_k)
                st.session_state.chat_history.append(
                    {"question": question, "answer": answer, "context": context}
                )
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # Clear button
    if st.session_state.chat_history:
        if st.button("🗑 Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
<div style="text-align:center; padding: 3rem 1rem; color: #4a4e5a;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
    <div style="font-family: 'DM Serif Display', serif; font-size: 1.4rem; color: #6a6460; margin-bottom: 0.5rem;">
        No document loaded yet
    </div>
    <div style="font-size: 0.9rem;">
        Enter your Groq API key and a PDF URL in the sidebar, then click <strong>Load & Index PDF</strong>
    </div>
</div>
""", unsafe_allow_html=True)
