import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re

st.title("My RAG App")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_chunk_pdf():
    reader = PdfReader("sample_base_knowledge.pdf")
    content = ""
    for page in reader.pages:
        content += page.extract_text()

    chunks = re.split(r'\.\s+', content)
    chunks = [c.strip() for c in chunks if c.strip() != ""]
    return chunks

@st.cache_resource
def build_collection(_model, chunks):
    embeddings = _model.encode(chunks)
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="my_docs")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

MODEL = load_model()
chunks = load_and_chunk_pdf()
collection = build_collection(MODEL, chunks)

query = st.text_input("Ask a question about your document:")

if query:
    results = collection.query(query_texts=[query], n_results=4)
    retrieved_chunks = results['documents'][0]
    context = "\n".join(retrieved_chunks)

    prompt = f"""Answer the question using only the context below.
Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write("### Answer")
    st.write(response['message']['content'])