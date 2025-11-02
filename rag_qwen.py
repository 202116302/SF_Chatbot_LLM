# ask.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama  # LangChain의 Ollama 래퍼
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DB_DIR   = "./chroma_db"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:7b"  # ✅ ollama에 pull된 모델 이름
TOP_K = 4                          # 검색 청크 개수 (비용/정확도 밸런스 포인트)

SYSTEM_PROMPT = """너는 스마트농업 교육을 지원하는 한국어 어시스턴트다.
반드시 아래 원칙을 지킨다.
1. 제공된 컨텍스트 안에 있는 내용만 말한다. 없으면 "자료에 없습니다"라고 말한다.
2. 상추 재배 방법은 단계별(파종→육묘→정식→환경관리→병해충→수확)로 설명한다.
3. 수치(온도, 습도, 차광, 관수, pF, kPa 등)는 원문과 다르게 바꾸지 않는다.
4. 답변 끝에 근거가 된 PDF 파일명과 페이지를 적는다.
5. 말투는 공적이고 전문적으로 한다.
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "다음은 관련 문서 조각들이야:\n"
        "{context}\n\n"
        "위의 내용만 근거로 다음 질문에 한국어로 답해줘:\n"
        "질문: {question}\n"
        "답변:"
    ),
)

def strip_think(text: str) -> str:
    # DeepSeek R1이 내놓는 <think> 태그 제거(가볍게 처리)
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def build_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})

def build_llm():
    # ChatOllama는 ollama 서버(기본 http://localhost:11434)에 접속합니다.
    # 필요한 경우 temperature, num_ctx, num_predict 등을 조절하세요.
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
        # num_ctx=4096,   # 문맥창 여유
        # num_predict=512
    )

def format_context(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        lines.append(f"[source: {src}, page: {page}] {d.page_content}")
    return "\n\n".join(lines[:TOP_K])

def answer(question: str):
    retriever = build_retriever()
    llm = build_llm()

    docs = retriever.get_relevant_documents(question)
    context = format_context(docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=QA_PROMPT.format(context=context, question=question)),
    ]
    raw = llm.invoke(messages).content
    text = strip_think(raw)

    print("🧠 답변:\n", text, "\n")
    print("📎 근거:")
    for d in docs:
        print("-", d.metadata.get("source"), "p.", d.metadata.get("page"))

if __name__ == "__main__":
    # 예시 질문
    answer("현재 누적광량이 8 mol m⁻² d⁻¹로 낮은 상태인데, 상추의 광포화점과 광보상점을 기준으로 보면 보광이 필요한가요?")
