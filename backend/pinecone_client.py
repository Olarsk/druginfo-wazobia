from pinecone import Pinecone
import os

class PineconeClient:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_host = os.getenv("PINECONE_INDEX_HOST")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(host=self.index_host)

    def search(self, query, top_k=3, namespace="default"):
        return self.index.search(
            namespace=namespace,
            query={"inputs": {"text": query}, "top_k": top_k},
            fields=["source", "text"]
        ) 