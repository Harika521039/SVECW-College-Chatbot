from dotenv import load_dotenv
import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Svecw College Chatbot", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

CSV_PATH = "svcew_details.csv"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs fine on CPU
TOP_K = 3                               # how many FAQ chunks to retrieve

# Two thresholds instead of one.
# score is cosine similarity in [-1, 1] (usually [0, 1] for this model).
HIGH_SIM_THRESHOLD = 0.65   # "this is basically the same question" -> return the stored answer as-is
LOW_SIM_THRESHOLD = 0.35    # "somewhat related" -> use retrieved rows as context for Gemini (RAG)
# below LOW_SIM_THRESHOLD -> question is out of scope for the FAQ -> plain Gemini answer, no context


# ---------- Data loading ----------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna("")
    df["Question"] = df["Question"].astype(str)
    df["Answer"] = df["Answer"].astype(str)
    return df


# ---------- Embedding model + FAISS index (built once, cached) ----------

@st.cache_resource
def load_embedder(model_name):
    return SentenceTransformer(model_name)


@st.cache_resource
def build_faiss_index(_embedder, questions):
    embeddings = _embedder.encode(
        list(questions), convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(embeddings)          # normalize so inner product == cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load the CSV file. Error: {e}")
    st.stop()

embedder = load_embedder(EMBED_MODEL_NAME)
index = build_faiss_index(embedder, df["Question"].tolist())

# Prefer an environment variable / Streamlit secret for the API key.
# Falls back to the original hardcoded key so the app still runs out of the box.
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini API Key not found. Please check your .env file.")
    st.stop()
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# ---------- Retrieval ----------

def retrieve(user_query, k=TOP_K):
    """Embed the query and fetch the k most similar FAQ rows via FAISS."""
    query_vec = embedder.encode([user_query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, idxs = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append(
            {
                "question": df.iloc[idx]["Question"],
                "answer": df.iloc[idx]["Answer"],
                "score": float(score),
            }
        )
    return results


# ---------- RAG generation ----------

def build_context(retrieved):
    blocks = [f"Q: {r['question']}\nA: {r['answer']}" for r in retrieved]
    return "\n\n".join(blocks)
def generate_rag_answer(user_query, retrieved):
    context = build_context(retrieved)

    prompt = f"""
You are an AI assistant for Shri Vishnu Engineering College for Women (SVECW).

Use the college FAQ information below as your primary source.

College FAQ:
{context}

Student Question:
{user_query}

Instructions:
1. If the answer is available in the college FAQ, answer using that information.
2. If the FAQ does not contain the answer, answer using your own general knowledge.
3. When using your own knowledge, clearly mention:
   "Note: This answer is based on general knowledge and not on the official SVECW FAQ database."
4. Give a clear, friendly, and concise answer.

Answer:
"""

    response = model.generate_content(prompt)
    return response.text


# ---------- UI ----------

st.title("Svecw College Chatbot")
st.write("Welcome to the College Chatbot! Ask me anything about the college.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved = retrieve(prompt, k=TOP_K)
    top_score = retrieved[0]["score"] if retrieved else 0.0
    
    try:
        if top_score >= HIGH_SIM_THRESHOLD:
            # Near-exact FAQ match -> return the stored answer directly (fast path, same as before)
            answer = retrieved[0]["answer"]
        elif top_score >= LOW_SIM_THRESHOLD:
            # Related but not identical -> RAG: let Gemini answer using retrieved rows as context
            answer = generate_rag_answer(prompt, retrieved)
        else:
            answer = f"""I couldn't find this information in the official SVECW FAQ database.

        {model.generate_content(prompt).text}

        Note: The above answer is generated using Gemini's general knowledge and may not represent official SVECW information."""
                
                
        

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    except Exception as e:
        st.error(f"Sorry, I couldn't generate a response. Error: {e}")