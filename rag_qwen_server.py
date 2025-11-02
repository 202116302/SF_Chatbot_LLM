# app.py
import os
import re
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ---------------- 설정 ----------------
DOCS_DIR = os.getenv("DOCS_DIR", "/docs")
DB_DIR = os.getenv("DB_DIR", "/data/chroma_db")
IMAGE_DIR = os.getenv("IMAGE_DIR", "/data/extracted_images")
TABLE_DIR = os.getenv("TABLE_DIR", "/data/extracted_tables")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = """너는 스마트농업·원예 교육용 한국어 어시스턴트다.
반드시 아래 원칙을 지킨다.
1. 제공된 컨텍스트 안에 있는 내용만 말한다. 없으면 없다고 말한다.
2. 상추 재배, 광환경, 보광, 차광, 육묘, 정식, 수분관리는 단계별로 설명한다.
3. 수치(온도, 광도, Lux, 시간, kPa, pF 등)는 원문과 다르게 바꾸지 않는다.
4. 답변 끝에 참조한 페이지/파일을 적는다.
5. 말투는 공적이고 전문적으로 한다.
"""

app = FastAPI(title="SmartAgri RAG API", version="1.0.0")

# ------ 공용 리소스(앱 시작 시 1회 로드) ------
embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

def build_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.2,
        # num_ctx=8192,  # 서버 GPU 여유되면 켠다
    )

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def format_context(docs) -> str:
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown.pdf")
        page = d.metadata.get("page", "?")
        lines.append(f"[source: {src}, page: {page}] {d.page_content}")
    return "\n\n".join(lines)

# ---------------- 모델 ----------------
class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]

# ---------------- 라우트 ----------------
@app.post("/rag/query", response_model=RAGResponse)
def rag_query(req: RAGRequest):
    # 1) 관련 문서 검색
    docs = retriever.get_relevant_documents(req.question)
    if not docs:
        return RAGResponse(
            answer="관련 문서를 찾지 못했습니다. PDF를 먼저 인덱싱해 주세요.",
            sources=[],
        )

    context = format_context(docs)
    llm = build_llm()

    human_prompt = (
        "다음은 상추 재배 관련 문서 조각이다.\n"
        f"{context}\n\n"
        "위 내용만 근거로 다음 질문에 답해라.\n"
        f"질문: {req.question}\n"
        "답변:"
    )

    msgs = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]
    raw = llm.invoke(msgs).content
    answer = strip_think(raw)

    used_sources = []
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        if src:
            used_sources.append(f"{src}#page={page}")

    return RAGResponse(answer=answer, sources=used_sources)

# ----- (옵션) PDF 업로드해서 다시 인덱싱 -----
@app.post("/rag/upload")
def rag_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF만 업로드 가능합니다.")
    save_path = os.path.join(DOCS_DIR, file.filename)
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(file.file.read())

    # 새로 들어온 PDF만 로드해서 추가 인덱싱
    loader = PyPDFLoader(save_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    vectordb.add_documents(chunks)
    vectordb.persist()

    return {"status": "ok", "message": f"{file.filename} 인덱싱 완료"}
