import streamlit as st
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

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
    return FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 2})

# ---------- LOAD LLM ----------
@st.cache_resource

def load_llm():
    gen = pipeline(
        "text-generation",
        model="google/flan-t5-small",
        max_new_tokens=120
    )
    return HuggingFacePipeline(pipeline=gen)


llm = load_llm()

# ---------- UI ----------
tweet = st.text_input("✍️ Enter Tweet")

if st.button("Analyze"):
    if tweet.strip() == "":
        st.warning("Please enter text")
    else:
        # ML prediction
        ml_pred = ml_model.predict([tweet])[0]
        ml_sentiment = "Positive" if ml_pred == 1 else "Negative"

        # RAG retrieval
        docs = retriever.invoke(tweet)
        context = "\n".join(d.page_content for d in docs)

        prompt = f"""
You are validating a sentiment classification.

ML Prediction: {ml_sentiment}

Similar labeled examples:
{context}

Tweet:
"{tweet}"

Return:
Sentiment:
Explanation:
Confidence:
"""

        final_result = llm.invoke(prompt)

        st.subheader("📊 Result")
        st.write(final_result)

