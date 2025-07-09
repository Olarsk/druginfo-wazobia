from .pinecone_client import PineconeClient
from models.gpt_model import chatgpt_generate

def rag_pipeline(user_query):
    pinecone_client = PineconeClient()
    retrieved = pinecone_client.search(user_query)
    context = " ".join([match["text"] for match in retrieved["matches"]])
    return chatgpt_generate(user_query, context) 