# Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_qwen_server.py index_pdf.py ./

# 우리가 쓸 데이터 경로
ENV DOCS_DIR=/docs
ENV DB_DIR=/data/chroma_db
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=qwen2.5:7b
ENV EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV TOP_K=4

# 👇 허깅페이스/트랜스포머 캐시를 우리가 쓸 수 있는 디렉터리로 고정
ENV HF_HOME=/data/hf_cache
ENV TRANSFORMERS_CACHE=/data/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/data/hf_cache

# 혹시 컨테이너 안에서만 돌릴 때도 권한 문제 없게 디렉터리 만들어두기
RUN mkdir -p /data/hf_cache && chmod -R 777 /data/hf_cache

EXPOSE 8000

CMD ["uvicorn", "rag_qwen_server:app", "--host", "0.0.0.0", "--port", "8000"]
