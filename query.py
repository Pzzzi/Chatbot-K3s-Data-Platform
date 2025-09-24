from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ new import (not deprecated)
import pprint

# Load embeddings (same as used when building vectorstore)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load persisted Chroma
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # top 5 chunks

def query_chunks(query: str, k: int = 5):
    """Retrieve top-k chunks for a given query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)

    output = []
    for i, doc in enumerate(results, 1):
        output.append({
            "rank": i,
            "chunk_preview": doc.page_content[:300] + "...",  # preview first 300 chars
            "metadata": doc.metadata,
        })
    return output

if __name__ == "__main__":
    # Example query
    query = "Tell me about Roberson-Murphy's financial performance"
    results = query_chunks(query, k=5)

    print(f"\n🔎 Query: {query}\n")
    pprint.pprint(results)
