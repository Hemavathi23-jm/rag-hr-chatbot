# 📄 HR-Policy RAG Chatbot (Local)

A **Retrieval-Augmented Generation (RAG) chatbot** built with **LangChain**, **ChromaDB**, and **Streamlit**, that allows you to query an **HR Policy PDF** locally — no API keys required.  
It supports **similarity search** and can optionally integrate with local LLMs such as **GPT4All** or **Ollama** for enhanced answers.

---

## 🚀 Features
- ✅ Runs **completely offline** – no API keys required  
- 📑 Indexes and stores HR policy documents using **ChromaDB**  
- 🔍 Supports **semantic similarity search** out-of-the-box  
- 🧠 Optional **local LLM integration** (GPT4All / Ollama) for more natural answers  
- 💬 **Conversation history** preserved across queries  
- ⚡ Simple **Streamlit interface**  

---

## 📂 Project Structure
├── index_docs.py # Script to index the HR policy PDF into a vector store
├── app_streamlit.py # Streamlit app for chatbot interface
├── HR-Policy (1).pdf # Example HR policy document (place in root directory)
└── vector_store/ # Chroma vector database (auto-generated after indexing)
