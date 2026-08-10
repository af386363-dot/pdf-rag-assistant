import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

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
    chunks = sent_tokenize(content)
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

@st.cache_resource
def build_bm25_index(chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25

def needs_retrieval(query):
    routing_prompt = f"""You are deciding whether a user's message requires searching a document to answer, 
or whether it can be answered directly without any document lookup (e.g. greetings, thanks, 
general chit-chat, or questions about the conversation itself rather than the document's content).

User message: {query}

Reply with exactly one word: RETRIEVE or DIRECT."""

    routing_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": routing_prompt}]
    )
    decision = routing_response['message']['content'].strip().upper()
    return "RETRIEVE" in decision

def generate_hypothetical_answer(query):
    hyde_prompt = f"""Write a short, plausible-sounding paragraph that could answer the following question. 
It's okay if some details are made up — this is just for search purposes.

Question: {query}

Hypothetical answer:"""

    hyde_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": hyde_prompt}]
    )
    return hyde_response['message']['content']

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
        "BM25 keyword search, HyDE query rewriting, reranking, agentic routing, and a local LLM (Ollama)."
    )
    st.write("**Pipeline:** Route → (Retrieve → Rerank →) Generate → (Verify)")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing your document..."):
        chunks = load_and_chunk_pdf(uploaded_file)
        collection = build_collection(MODEL, chunks)
        bm25 = build_bm25_index(chunks)

    st.success(f"Document indexed — {len(chunks)} chunks ready to search.")

    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Deciding how to answer..."):
            should_retrieve = needs_retrieval(query)

        if not should_retrieve:
            with st.spinner("Generating a direct answer..."):
                direct_prompt = f"Answer the following message naturally and briefly: {query}"
                response = ollama.chat(
                    model="llama3.2",
                    messages=[{"role": "user", "content": direct_prompt}]
                )
            st.markdown("### Answer")
            st.write(response['message']['content'])
            st.caption("No document retrieval was needed for this message.")

        else:
            with st.spinner("Searching and generating an answer..."):
                hypothetical_answer = generate_hypothetical_answer(query)

                semantic_results = collection.query(query_texts=[hypothetical_answer], n_results=10)
                semantic_chunks = semantic_results['documents'][0]

                tokenized_query = query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
                bm25_chunks = [chunks[i] for i in top_bm25_indices]

                combined_chunks = list(set(semantic_chunks + bm25_chunks))

                pairs = [[query, chunk] for chunk in combined_chunks]
                scores = RERANKER.predict(pairs)

                scored_chunks = list(zip(scores, combined_chunks))
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

            with st.expander("🧠 Hypothetical answer used for search (HyDE)"):
                st.write(hypothetical_answer)

            with st.expander("🔍 Faithfulness check"):
                st.write(faithfulness_result)

            with st.expander("📄 Retrieved context (used to generate this answer)"):
                for i, chunk in enumerate(top_chunks):
                    st.markdown(f"**Chunk {i+1}:** {chunk}")