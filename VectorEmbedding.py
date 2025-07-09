import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm
from pinecone import Pinecone
import time

# Load text files
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Load BNF & EMDEX
bnf_text = load_text("data/formatted_BNF.txt")
emdex_text = load_text("data/formatted_EMDEX.txt")
pthb9_text = load_text("data/formatted_PTHB9.txt")

# Text splitter for better searchability
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

# Split each document into chunks
bnf_chunks = text_splitter.split_text(bnf_text)
emdex_chunks = text_splitter.split_text(emdex_text)
pthb9_chunks = text_splitter.split_text(pthb9_text)

print(f"🔹 BNF Chunks: {len(bnf_chunks)} | 🔹 EMDEX Chunks: {len(emdex_chunks)} | 🔹 PTHB9 Chunks: {len(pthb9_chunks)}")

# Load env variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# ✅ Connect to Pinecone index (latest SDK)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Build records with 'values' field as a list (for integrated embedding, 'values': [text])
def make_vectors(chunks, source):
    return [(f"{source}_{i}", chunk, {'source': source}) for i, chunk in enumerate(chunks)]

vectors = (
    make_vectors(bnf_chunks, "BNF-84") +
    make_vectors(emdex_chunks, "EMDEX") +
    make_vectors(pthb9_chunks, "PTHB9")
)

# Batch upsert with 'values' as a list (text)
batch_size = 96
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    records = [{"_id": id_, "text": text, **meta} for id_, text, meta in batch]
    index.upsert_records(records=records, namespace="default")
    time.sleep(10)
       
print("✅ BNF-84, EMDEX, and PTHB9 Data Stored in Pinecone!")