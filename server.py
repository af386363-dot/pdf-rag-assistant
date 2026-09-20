from flask import Flask, render_template, request, jsonify
import chromadb
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

print("Loading models... this may take a moment.")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Models loaded.")

# Global state for the currently loaded document
state = {
    "chunks": None,
    "collection": None,
    "bm25": None,
    "chat_history": []
}

client = chromadb.Client()


def load_and_chunk_pdf(file):
    reader = PdfReader(file)
    content = ""
    for page in reader.pages:
        content += page.extract_text()
    chunks = sent_tokenize(content)
    chunks = [c.strip() for c in chunks if c.strip() != ""]
    return chunks


def build_collection(chunks):
    embeddings = MODEL.encode(chunks)
    try:
        client.delete_collection(name="my_docs")
    except Exception:
        pass
    collection = client.create_collection(name="my_docs")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection


def build_bm25_index(chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


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


def resolve_query(query, history):
    if not history:
        return query

    history_text = ""
    for turn in history[-3:]:
        history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

    resolve_prompt = f"""Given this recent conversation history:

{history_text}

Rewrite the following follow-up question so it's fully self-contained and clear on its own, 
resolving any pronouns or references to the conversation above. If it's already clear, return it unchanged.

Follow-up question: {query}

Rewritten question:"""

    resolve_response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": resolve_prompt}]
    )
    return resolve_response['message']['content'].strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    chunks = load_and_chunk_pdf(file)
    collection = build_collection(chunks)
    bm25 = build_bm25_index(chunks)

    state["chunks"] = chunks
    state["collection"] = collection
    state["bm25"] = bm25
    state["chat_history"] = []

    return jsonify({"message": f"Document indexed — {len(chunks)} chunks ready to search."})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data["question"]

    resolved_query = resolve_query(query, state["chat_history"])
    should_retrieve = needs_retrieval(resolved_query)

    if not should_retrieve:
        direct_prompt = f"Answer the following message naturally and briefly: {query}"
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": direct_prompt}]
        )
        answer = response['message']['content']
        state["chat_history"].append({"question": query, "answer": answer})
        return jsonify({"answer": answer, "used_retrieval": False})

    chunks = state["chunks"]
    collection = state["collection"]
    bm25 = state["bm25"]

    hypothetical_answer = generate_hypothetical_answer(resolved_query)

    semantic_results = collection.query(query_texts=[hypothetical_answer], n_results=10)
    semantic_chunks = semantic_results['documents'][0]

    tokenized_query = resolved_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_chunks = [chunks[i] for i in top_bm25_indices]

    combined_chunks = list(set(semantic_chunks + bm25_chunks))

    pairs = [[resolved_query, chunk] for chunk in combined_chunks]
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
    answer = response['message']['content']

    faithfulness_result = check_faithfulness_llm(answer, context)

    state["chat_history"].append({"question": query, "answer": answer})

    return jsonify({
        "answer": answer,
        "used_retrieval": True,
        "resolved_query": resolved_query,
        "hypothetical_answer": hypothetical_answer,
        "faithfulness": faithfulness_result,
        "top_chunks": top_chunks
    })


if __name__ == "__main__":
    app.run(debug=True)