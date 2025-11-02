# Dockerfile
FROM python:3.11-slim

# 시스템 패키지 조금만
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드
COPY rag_qwen_server.py index_pdf.py ./

# 데이터는 호스트에서 마운트할 거라 컨테이너 안에서만 경로 정해둠
ENV DOCS_DIR=/docs
ENV DB_DIR=/data/chroma_db
# Ollama 컨테이너와 네트워크로 붙일 거라 이름으로 접근
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=qwen2.5:7b
ENV EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV TOP_K=4

# 컨테이너가 뜨면 일단 FastAPI만 올린다
CMD ["uvicorn", "rag_qwen_server:app", "--host", "0.0.0.0", "--port", "8000"]
