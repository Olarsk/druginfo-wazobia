from .pinecone_client import PineconeClient
from models.gpt_model import chatgpt_generate

def rag_pipeline(user_query):
    pinecone_client = PineconeClient()
    retrieved = pinecone_client.search(user_query)
    
    # Pass the raw Pinecone RAG output directly to the LLM
    return chatgpt_generate(user_query, str(retrieved))