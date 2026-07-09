import chromadb
import ollama
from sentence_transformers import SentenceTransformer
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
from pypdf import PdfReader

reader = PdfReader("sample_base_knowledge.pdf")

content = ""
for page in reader.pages:
    content += page.extract_text()

print(content)
print(f"Length of content: {len(content)}")
import re
chunks = re.split(r'\.\s+', content)
chunks = [c.strip() for c in chunks if c.strip() != ""]
for i, c in enumerate(chunks):
    print(f"Chunk {i}: {c}")

embeddings=MODEL.encode(chunks)
print(embeddings.shape)

client=chromadb.Client()
collection = client.get_or_create_collection(name="my_docs")
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print("Stored", collection.count(), "chunks in the database")

query = "How do vaccines create immunity?"

results = collection.query(query_texts=[query], n_results=4)
)

print(results)
retrieved_chunks=results['documents'][0]
context="\n".join(retrieved_chunks)

prompt = f"""Answer the question using only the context below.
Context:
{context}

Question: {query}

Answer:"""

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}]
)

print(response['message']['content'])

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

faithfulness_result = check_faithfulness_llm(response['message']['content'], context)
print("Faithfulness check:")
print(faithfulness_result)