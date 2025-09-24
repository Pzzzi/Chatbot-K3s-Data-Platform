from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatGroq(model="moonshotai/kimi-k2-instruct", api_key=os.environ["GROQ_API_KEY"], temperature=0.7)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

def ask_question_rag(question: str):
    """Run RAG pipeline and return both answer + sources."""
    result = qa_chain.__call__(question, return_only_outputs=False)
    return {
        "answer": result.get("result", result.get("output_text", "")),
        "sources": [doc.metadata for doc in result.get("source_documents", [])]
    }
