from langchain_groq import ChatGroq
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///Dataset/data/finance.db", include_tables=["sectors", "deals", "companies"])

llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.7,
    api_key=os.environ["GROQ_API_KEY"]
)

db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    verbose=True, 
    use_query_checker=True,
    top_k=20
)

def ask_question(prompt: str):
    """Run query against LLM and return answer + sources consistently."""
    result = db_chain.invoke({"query": prompt})  # your current chain

    # Wrap outputs consistently
    answer = result.get("result") or result.get("output_text") or ""
    sources = result.get("source_documents") or []

    # Convert sources to metadata if they are LangChain Document objects
    formatted_sources = []
    for doc in sources:
        if hasattr(doc, "metadata"):
            formatted_sources.append(doc.metadata)
        else:
            formatted_sources.append(doc)

    return {
        "answer": answer,
        "sources": formatted_sources
    }