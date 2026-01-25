from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

with open("rag_dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.create_documents([text])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)
db.save_local("vector_db")

print("✅ Dataset connected to RAG successfully")
