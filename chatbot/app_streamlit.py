"""
Streamlit + LangChain conversational RAG app.
Runs completely locally without API keys or .env file.
"""

import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory       
from langchain.chains import RetrievalQA

# Config
PERSIST_DIR = "vector_store"
PAGE_TITLE = "HR-Policy RAG Chatbot (Local)"
PAGE_ICON = "📄"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

st.title("HR-Policy RAG Chatbot")
st.markdown("Ask questions about the HR Policy PDF. Running completely locally.")

# Try to use a local LLM, fallback to simple similarity search
@st.cache_resource
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        st.success("✅ Vector database loaded successfully")
        return vectordb
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        st.info("Please run index_docs.py first to create the vector store")
        return None

vectordb = load_vectorstore()
if vectordb is None:
    st.stop()

# Setup retriever
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Simple similarity-based QA (no LLM required)
def simple_qa_search(question):
    """Simple QA using similarity search without LLM"""
    try:
        # Get similar documents
        docs = retriever.get_relevant_documents(question)
        
        if not docs:
            return "I couldn't find relevant information in the HR policy document.", []
        
        # Combine the most relevant chunks
        context = "\n\n".join([f"Document {i+1} (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" 
                              for i, doc in enumerate(docs[:3])])
        
        # Simple template-based response
        response = f"""Based on the HR policy document, I found this relevant information:

{context}

This information appears relevant to your question about: {question}

[Note: This is a similarity-based search. For more nuanced answers, consider setting up a local LLM.]"""
        
        return response, docs
        
    except Exception as e:
        return f"Error searching documents: {str(e)}", []

# Enhanced version with local LLM if available
@st.cache_resource
def try_load_local_llm():
    """Try to load a local LLM, return None if not available"""
    try:
        # Try GPT4All first
        try:
            from langchain.llms import GPT4All
            llm = GPT4All(
                model="orca-mini-3b.ggmlv3.q4_0.bin",
                temp=0.1,
                verbose=False
            )
            st.success("✅ Local LLM (GPT4All) loaded successfully")
            return llm
        except ImportError:
            pass
        
        # Try Ollama as fallback
        try:
            from langchain.llms import Ollama
            llm = Ollama(model="llama2", temperature=0.1)
            st.success("✅ Local LLM (Ollama) loaded successfully")
            return llm
        except ImportError:
            pass
            
    except Exception as e:
        st.warning(f"Local LLM not available: {e}")
    
    return None

# Try to load local LLM
llm = try_load_local_llm()

if llm:
    # Use LLM-enhanced QA if available
    prompt_template = """Use the following context to answer the question. 
    If the answer isn't in the context, say you don't know.

    Context: {context}

    Question: {question}

    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    def query_bot(user_input):
        try:
            result = qa_chain({"query": user_input})
            return result["result"], result.get("source_documents", [])
        except Exception as e:
            return f"LLM Error: {str(e)}", []
else:
    # Use simple similarity search
    st.info("🔍 Using similarity search (no local LLM detected). Install GPT4All or Ollama for better answers.")
    query_bot = simple_qa_search

# UI
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Your question about the HR policy", "")
    submitted = st.form_submit_button("Ask")
    
    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "text": user_input})
        with st.spinner("Searching HR policy..."):
            answer, srcs = query_bot(user_input)
        
        st.session_state.messages.append({"role": "assistant", "text": answer})
        
        # Show answer
        st.write("**Answer:**")
        st.info(answer)
        
        # Show sources
        if srcs:
            st.write("**Sources:**")
            for i, doc in enumerate(srcs[:3]):
                meta = getattr(doc, "metadata", {}) or {}
                source = meta.get("source", "Unknown")
                page = meta.get("page", "N/A")
                st.write(f"{i+1}. {source} (Page {page})")
                # Show small excerpt
                st.write(f"   Excerpt: {doc.page_content[:150]}...")

# Conversation history
if st.session_state.messages:
    st.write("---")
    st.write("### Conversation History")
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['text']}")
        else:
            st.markdown(f"**Bot:** {m['text']}")

st.markdown("---")
st.caption("Running completely locally - No API keys required")
st.caption("To improve answers: pip install gpt4all")