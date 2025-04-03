import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data_path):
    """Load all PDF files from a directory using PyMuPDFLoader."""
    try:
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        if not documents:
            print("⚠️ No PDFs found in the directory!")
        return documents
    except Exception as e:
        print(f"❌ Error loading PDFs: {e}")
        return []

documents = load_pdf_files(DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data, chunk_size=500, chunk_overlap=50):
    """Split documents into manageable text chunks."""
    if not extracted_data:
        print("⚠️ No data to split into chunks!")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(extracted_data)

text_chunks = create_chunks(documents)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    """Load the HuggingFace Embedding Model."""
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

def store_embeddings(text_chunks, model, db_path):
    """Store text embeddings using FAISS."""
    if not text_chunks or model is None:
        print("⚠️ No text chunks or embedding model found. Skipping FAISS storage.")
        return
    
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)

    try:
        db = FAISS.from_documents(text_chunks, model)
        db.save_local(db_path)
        print(f"✅ FAISS database saved at: {db_path}")
    except Exception as e:
        print(f"❌ Error saving FAISS database: {e}")

store_embeddings(text_chunks, embedding_model, DB_FAISS_PATH)
