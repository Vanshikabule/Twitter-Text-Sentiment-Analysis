import streamlit as st
import pickle

from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered"
)

# ---------- LOAD ML MODEL ----------
@st.cache_resource
def load_ml_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

ml_model = load_ml_model()

# ---------- LOAD VECTOR DB ----------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        folder_path="vector_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 2})

# ---------- LOAD LLM ----------
@st.cache_resource
def load_llm():
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=120
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm

llm = load_llm()

# ---------- UI ----------
st.title("🐦 Twitter Text Sentiment Analysis")
st.write("Hybrid ML + RAG + LLM sentiment validation")

tweet = st.text_area("✍️ Enter Tweet", height=100)

if st.button("Analyze"):
    if not tweet.strip():
        st.warning("Please enter some text")
    else:
        # ----- Traditional ML Prediction -----
        ml_pred = ml_model.predict([tweet])[0]
        ml_sentiment = "Positive" if ml_pred == 1 else "Negative"

        # ----- Retrieve Similar Examples -----
        docs = retriever.invoke(tweet)
        context = "\n".join(d.page_content for d in docs)

        # ----- LLM Prompt -----
        prompt = f"""
Classify the sentiment of the tweet using the examples below.

ML Prediction: {ml_sentiment}

Similar labeled examples:
{context}

Tweet:
{tweet}

Respond ONLY in this format:

Sentiment:
Explanation:
Confidence:
"""

        # ----- LLM Inference -----
        result = llm.invoke(prompt)

        # ----- Display Result -----
        st.subheader("📊 Result")
        st.markdown(result)
