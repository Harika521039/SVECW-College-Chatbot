# 🎓 SVECW College Information Chatbot

AI-powered chatbot that answers queries about Shri Vishnu Engineering College for Women using NLP and Information Retrieval.

### 🔧 Tech Stack
Python, Streamlit, Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS, Pandas, NumPy, Retrieval-Augmented Generation (RAG), Google Gemini 1.5 Flash API

### 📌 Problem Statement
Students struggle to find information about admissions, fees, hostel, and placements scattered across college websites. This chatbot provides instant answers to natural language queries.

### 💡 Solution Approach
1. **Dataset Creation**: Compiled 127 Q&A pairs from official college resources into `svcew_details.csv`
2. **Embedding**: Each FAQ question is encoded into a dense semantic vector using the `all-MiniLM-L6-v2` Sentence-Transformer model
3. **Vector Search**: Embeddings are indexed with `faiss.IndexFlatIP` for fast cosine-similarity search over the FAQ set
4. **Hybrid Threshold Routing**:
   - **High similarity (≥ 0.65)** → return the matched FAQ answer directly (fast path)
   - **Medium similarity (0.35–0.65)** → **RAG**: the top-3 retrieved FAQ rows are passed to Gemini as context, and Gemini generates a grounded answer from them
   - **Low similarity (< 0.35)** → question is out of scope for the FAQ → Gemini answers from general knowledge
5. **Frontend**: Interactive UI built with Streamlit for real-time chat

### 🚀 Features
- Answers 127+ college FAQs instantly
- Semantic search (Sentence-Transformers + FAISS) instead of keyword-based TF-IDF, so it understands paraphrased/reworded questions
- Retrieval-Augmented Generation: Gemini answers are grounded in retrieved FAQ context instead of guessing
- Gemini fallback for genuinely out-of-scope questions
- Streamlit web interface with chat history

### ▶️ How to Run
```bash
git clone https://github.com/Harika521039/college-chatbot.git
cd college-chatbot
pip install -r requirements.txt
streamlit run college.py