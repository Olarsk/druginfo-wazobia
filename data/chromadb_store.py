from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Load embeddings
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Path to ChromaDB storage
chroma_db_path = "./chroma_db"

def load_data_to_chromadb(bnf_chunks, emdex_chunks, pthb9_chunks):
    """Store BNF, EMDEX, and PTHB9 data in ChromaDB."""

    bnf_docs = [Document(page_content=chunk, metadata={"source": "BNF-84"}) for chunk in bnf_chunks]
    emdex_docs = [Document(page_content=chunk, metadata={"source": "EMDEX"}) for chunk in emdex_chunks]
    pthb9_docs = [Document(page_content=chunk, metadata={"source": "PTHB9"}) for chunk in pthb9_chunks]

    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=bnf_docs + emdex_docs + pthb9_docs,
        embedding=embedding_model,
        persist_directory=chroma_db_path,
    )

    print("✅ BNF-84, EMDEX, and PTHB9 Data Stored in ChromaDB!")

def retrieve_drug_info(query, top_k=3):
    """Retrieve top 3 results from each dataset in ChromaDB."""
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)

    results = []
    for source in ["BNF-84", "EMDEX", "PTHB9"]:
        source_results = vectorstore.similarity_search(query, k=top_k, filter={"source": source})
        results.extend(source_results)

    retrieved_texts = {source: [] for source in ["BNF-84", "EMDEX", "PTHB9"]}
    for res in results:
        retrieved_texts[res.metadata["source"]].append(res.page_content)

    summarized_info = "\n\n".join([
        f"Source: {src}\n" + "\n".join(retrieved_texts[src])
        for src in retrieved_texts if retrieved_texts[src]
    ])

    return summarized_info
