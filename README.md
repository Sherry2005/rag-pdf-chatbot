# High-Performance Financial RAG Pipeline (PDF Chatbot)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://github.com/Sherry2005/rag-pdf-chatbot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An optimized Retrieval-Augmented Generation (RAG) system designed for high-speed semantic search and conversational analysis over dense technical and financial document sets. This project leverages **LangChain**, **FAISS**, and the **Groq Llama 3** inference engine to provide near-instantaneous responses even with complex PDF inputs.

## 🚀 Key Features

*   **Lightning-Fast Inference:** Integrated with **Groq API** (Llama 3.3/70B) for sub-second token generation.
*   **Optimized Retrieval:** Uses **FAISS (Facebook AI Similarity Search)** for efficient vector indexing and similarity search.
*   **Semantic Understanding:** Employs **HuggingFace** sentence-transformers to generate high-quality document embeddings.
*   **Production-Grade RAG:** Implements recursive character text splitting and context-aware prompt templates to minimize hallucinations.
*   **Interactive UI:** A clean, tabbed **Streamlit** interface featuring chat history, document previewing, and reasoning traces.

## 🛠️ Tech Stack

*   **Orchestration:** [LangChain](https://github.com/langchain-ai/langchain)
*   **LLM Engine:** [Groq](https://groq.com/) (Llama 3.3 70B)
*   **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
*   **Embeddings:** [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **PDF Parsing:** `pdfplumber` / `PyPDF2`

## 🏗️ Architecture

1.  **Ingestion:** PDF files are parsed and cleaned.
2.  **Chunking:** Documents are split into overlapping segments using `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Text chunks are transformed into 384-dimensional vectors.
4.  **Indexing:** Vectors are stored in a local FAISS index for high-speed top-k retrieval.
5.  **Query Loop:**
    *   User query is embedded.
    *   Context is retrieved from FAISS.
    *   Prompt is augmented with context and chat history.
    *   Groq delivers the final synthesized answer.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sherry2005/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Performance Benchmarks

| Feature | Performance |
| :--- | :--- |
| **Embedding Generation** | ~200ms (Small-Medium PDFs) |
| **Search Latency** | <50ms (FAISS) |
| **LLM Time-to-First-Token** | ~0.15s (via Groq) |

## 🛡️ Reliability & Safety

*   **Context Grounding:** The system is strictly instructed to answer only based on provided context to prevent "hallucination."
*   **Rate Limit Management:** Implements graceful backoff for API calls.
*   **Memory Management:** FAISS index is cleared/rebuilt per session to ensure privacy and accuracy across different document sets.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
**Developed by [Sherry Mohareb](https://github.com/Sherry2005)**
*Building scalable AI systems for a safer and more efficient future.*
