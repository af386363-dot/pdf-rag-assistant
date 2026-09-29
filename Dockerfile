FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
# CPU-only torch keeps the image much smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake models + nltk data into the image so nothing downloads at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" \
 && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .
EXPOSE 8080
# 1 worker on purpose: the app keeps the loaded PDF in global memory
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "300", "-b", "0.0.0.0:8080", "server:app"]
