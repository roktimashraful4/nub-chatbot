# ingest/ingest.py
import os
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
os.makedirs(VECTOR_DIR, exist_ok=True)

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1) Load text files
paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
docs = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read()
    docs.append(Document(page_content=txt, metadata={"source": os.path.basename(p)}))

print(f"✅ Loaded {len(docs)} files.")

# 2) Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks.")

# 3) Create embeddings with HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4) Build FAISS vectorstore
db = FAISS.from_documents(chunks, embeddings)
db.save_local(VECTOR_DIR)
print(f"✅ Saved FAISS vectorstore to: {VECTOR_DIR}")