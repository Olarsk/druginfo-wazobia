from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

print("Pinecone version:", Pinecone.__module__)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")  # Add this to your .env or set manually

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(host=pinecone_index_host)

query_text = "What is the dosage for paracetamol?"

results = index.search(
    namespace="default",
    query={
        "inputs": {"text": query_text},
        "top_k": 3
    },
    fields=["source", "text"]  # Adjust fields as needed
)

print(results)