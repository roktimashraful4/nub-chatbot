# ingest/ingest.py
import os
import glob
from dotenv import load_dotenv
import sys

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

print("=" * 80)
print("🔄 STARTING VECTOR STORE INGESTION")
print("=" * 80)

# 1) Load text files
print(f"\n📂 Loading files from: {DATA_DIR}")
paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))

if not paths:
    print("❌ ERROR: No .txt files found in data directory!")
    sys.exit(1)

print(f"✅ Found {len(paths)} .txt files to process:\n")

docs = []
failed_files = []

for i, p in enumerate(paths, 1):
    try:
        filename = os.path.basename(p)
        file_size = os.path.getsize(p) / 1024  # Size in KB
        
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
        
        if txt.strip():  # Only add if file has content
            docs.append(Document(page_content=txt, metadata={"source": filename}))
            print(f"   {i:2d}. ✓ {filename:60s} ({file_size:>8.1f} KB)")
        else:
            print(f"   {i:2d}. ⚠ {filename:60s} (EMPTY - skipped)")
            
    except Exception as e:
        print(f"   {i:2d}. ✗ {os.path.basename(p):60s} (ERROR: {str(e)})")
        failed_files.append((os.path.basename(p), str(e)))

print(f"\n✅ Successfully loaded {len(docs)} files")
if failed_files:
    print(f"⚠️  Failed to load {len(failed_files)} files:")
    for filename, error in failed_files:
        print(f"   - {filename}: {error}")

if not docs:
    print("❌ ERROR: No documents to process!")
    sys.exit(1)

# 2) Split into chunks
print(f"\n🔪 Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks (avg {len(chunks)//len(docs):.0f} chunks per file)")

# 3) Create embeddings with HuggingFace
print(f"\n🧠 Creating embeddings using HuggingFace...")
print(f"   Model: sentence-transformers/all-MiniLM-L6-v2")
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"✅ Embeddings model loaded successfully")
except Exception as e:
    print(f"❌ ERROR loading embeddings: {str(e)}")
    sys.exit(1)

# 4) Build FAISS vectorstore
print(f"\n🏗️  Building FAISS vectorstore...")
try:
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)
    print(f"✅ FAISS vectorstore created successfully")
except Exception as e:
    print(f"❌ ERROR creating vectorstore: {str(e)}")
    sys.exit(1)

# Verification
print(f"\n📊 INGESTION SUMMARY")
print("=" * 80)
print(f"✅ Files processed:        {len(docs)}")
print(f"✅ Total chunks created:   {len(chunks)}")
print(f"✅ Vectorstore location:   {VECTOR_DIR}")
print(f"✅ Embeddings model:       sentence-transformers/all-MiniLM-L6-v2")

if failed_files:
    print(f"⚠️  Files with errors:      {len(failed_files)}")

# Check if vectorstore files were created
index_file = os.path.join(VECTOR_DIR, "index.faiss")
if os.path.exists(index_file):
    index_size = os.path.getsize(index_file) / (1024 * 1024)  # Size in MB
    print(f"✅ Index file size:        {index_size:.2f} MB")
else:
    print(f"❌ WARNING: Index file not found at {index_file}")

print("=" * 80)
print("✅ VECTORSTORE INGESTION COMPLETE!")
print("=" * 80)