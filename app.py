import streamlit as st
import sqlite3
import pandas as pd
from chat_rag import ask_question_rag

# ---- DB connection ----
DB_PATH = "Dataset/data/finance.db"

def run_query(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ---- Streamlit UI ----
st.set_page_config(page_title="K3s AI Data Chatbot", layout="wide")

st.sidebar.header("Data Explorer")
table_choice = st.sidebar.selectbox("Choose a table", ["companies", "deals", "sectors"])
if st.sidebar.button("View Table"):
    df = run_query(f"SELECT * FROM {table_choice} LIMIT 20;")
    st.sidebar.write(df)

st.title("💬 K3s AI Data Chatbot")
st.write("Ask questions about companies, deals, and sectors using natural language.")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your question about the data..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question_rag(prompt)
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            # Optional: display sources
            if sources:
                with st.expander("📄 Sources"):
                    st.json(sources)

    st.session_state["messages"].append({"role": "assistant", "content": answer})

