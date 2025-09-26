"""
Index the uploaded HR policy PDF into a vector store.
Runs completely locally without API keys.
"""

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = "HR-Policy (1).pdf"  # Make sure this file is in the same directory
PERSIST_DIR = "vector_store"

def load_and_split(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please make sure 'HR-Policy (1).pdf' is in the same directory as this script")
        return None
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # split into chunks ~ 500 tokens (or ~1000 chars)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} source pages -> {len(split_docs)} text chunks")
    return split_docs

def main():
    docs = load_and_split(PDF_PATH)
    if docs is None:
        return
    
    # Always use local embeddings
    print("Using local sentence-transformers embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use Chroma (on-disk persistence)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(f"Vector store persisted to {PERSIST_DIR}")
    print("Indexing complete! You can now run the Streamlit app.")

if __name__ == "__main__":
    main()