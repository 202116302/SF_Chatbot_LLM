# ingest.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR = "/srv/docs"
DB_DIR = "/srv/chroma_db"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(dir_path: str):
    paths = list(Path(dir_path).rglob("*.pdf")) + list(Path(dir_path).rglob("*.txt"))
    docs = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(p)).load()
        else:
            docs += TextLoader(str(p), encoding="utf-8").load()
    return docs

def main():
    docs = load_documents(DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vectordb.persist()
    print(f"✅ Indexed {len(chunks)} chunks into {DB_DIR}")

if __name__ == "__main__":
    main()
