import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import re

st.set_page_config(page_title="RAG App", page_icon="📄", layout="centered")

st.title("📄 Document Q&A")
st.caption("Upload a PDF and ask questions about it — powered by a local RAG pipeline.")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_data
def load_and_chunk_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
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

def check_faithfulness_llm(answer, context):
    eval_prompt = f"""You are checking whether an answer is fully supported by a given context.

Context:
{context}

Answer to check:
{answer}

Does every claim in the answer follow directly from the context, with no invented facts? 
Reply with exactly one word: YES or NO.
If NO, on a second line, briefly state which part is not supported."""

    eval_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": eval_prompt}]
    )
    return eval_response['message']['content']

with st.spinner("Loading models..."):
    MODEL = load_model()
    RERANKER = load_reranker()

with st.sidebar:
    st.header("About")
    st.write(
        "This app answers questions about a PDF you upload, using local embeddings, "
        "a vector database, reranking, and a local LLM (Ollama)."
    )
    st.write("**Pipeline:** Load → Chunk → Embed → Store → Retrieve → Rerank → Generate → Verify")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing your document..."):
        chunks = load_and_chunk_pdf(uploaded_file)
        collection = build_collection(MODEL, chunks)

    st.success(f"Document indexed — {len(chunks)} chunks ready to search.")

    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Searching and generating an answer..."):
            results = collection.query(query_texts=[query], n_results=10)
            retrieved_chunks = results['documents'][0]

            pairs = [[query, chunk] for chunk in retrieved_chunks]
            scores = RERANKER.predict(pairs)

            scored_chunks = list(zip(scores, retrieved_chunks))
            scored_chunks.sort(reverse=True, key=lambda x: x[0])

            top_chunks = [chunk for score, chunk in scored_chunks[:6]]
            context = "\n".join(top_chunks)

            prompt = f"""Answer the question using only the context below.
Context:
{context}

Question: {query}

Answer:"""

            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}]
            )

            faithfulness_result = check_faithfulness_llm(response['message']['content'], context)

        st.markdown("### Answer")
        st.write(response['message']['content'])

        with st.expander("🔍 Faithfulness check"):
            st.write(faithfulness_result)

        with st.expander("📄 Retrieved context (used to generate this answer)"):
            for i, chunk in enumerate(top_chunks):
               st.markdown(f"**Chunk {i+1}:** {chunk}")