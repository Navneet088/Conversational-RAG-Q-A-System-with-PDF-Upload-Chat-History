# 📄 Conversational RAG Q&A with PDF Upload & Chat History

A Streamlit-based interactive application that enables users to upload PDF files and conduct conversational question-answering using Retrieval-Augmented Generation (RAG). The system uses embeddings and a Groq LLM to answer questions grounded in the document content, while maintaining chat history for context-aware conversation.

---

## 🧠 Features

✔ Upload one or more PDF documents  
✔ Extract and split PDF text into chunks  
✔ Create semantic embeddings (HuggingFace)  
✔ Store and search using Chroma vector store  
✔ History-aware retrieval for context-sensitive questions  
✔ Chat with the content using an LLM (Groq)  
✔ View a persistent chat history per session  
✔ Simple Streamlit UI

---

## 🚀 Screenshots

> Add your screenshots here (optional)  
> Example:
> ![app screenshot](./screenshots/app_ui.png)

---

## 📦 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Document Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace (`all-miniLM-L6-v2`) |
| Vector Store | Chroma |
| LLM | Groq (Llama 3.1) |
| RAG Orchestration | LangChain |

---

## 🧠 How It Works

1. **Upload PDFs** — Users upload one or more PDF files.  
2. **Text Splitting** — PDFs are split into overlapping chunks to build context segments.  
3. **Vector Embeddings** — Each chunk is embedded using HuggingFace semantic embeddings.  
4. **RAG Retrieval** — A retriever uses the embeddings to find the most relevant chunks for each question.  
5. **Contextual QA** — A Groq LLM answers user questions grounded in the retrieved content.  
6. **Chat History** — Conversation history is maintained per session for follow-ups.

---

## 📥 Requirements

- Python 3.10+  
- A valid **Groq API Key**  
- Install dependencies:

```bash
pip install -r requirements.txt
