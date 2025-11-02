# index_pdfs_advanced.py
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import os

# Camelot 설치 여부 확인
try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("⚠️ Camelot이 설치되지 않았습니다. 기본 표 추출 모드로 실행됩니다.")
    print("   정확한 표 추출을 원하시면: pip install camelot-py[cv] opencv-python")


DOCS_DIR = "./docs"
DB_DIR = "./data/chroma_db"
IMAGE_DIR = "./data/extracted_images"
TABLE_DIR = "./data/extracted_tables"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# dir_list = [DOCS_DIR, DB_DIR, IMAGE_DIR, TABLE_DIR]
#
# for i in dir_list:
#     if not os.path.exists(i):
#         os.makedirs(i)
#

# 설정
MIN_IMAGE_SIZE = 10000  # 10KB 이하 이미지는 무시 (아이콘 등 제외)
EXTRACT_TABLES = True  # 표 추출 활성화
EXTRACT_IMAGES = True  # 이미지 추출 활성화


def extract_images_from_pdf(pdf_path):
    """PDF에서 이미지 추출 및 저장"""
    if not EXTRACT_IMAGES:
        return []

    doc = fitz.open(pdf_path)
    images_info = []

    Path(IMAGE_DIR).mkdir(exist_ok=True)
    pdf_name = Path(pdf_path).stem

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # 너무 작은 이미지 필터링 (아이콘, 로고 등)
                if len(image_bytes) < MIN_IMAGE_SIZE:
                    continue

                # 이미지 저장
                image_filename = f"{pdf_name}_p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                image_path = Path(IMAGE_DIR) / image_filename

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # 이미지 크기 정보 추가
                try:
                    with Image.open(io.BytesIO(image_bytes)) as pil_img:
                        width, height = pil_img.size
                except:
                    width, height = None, None

                images_info.append({
                    "page": page_num + 1,
                    "path": str(image_path),
                    "filename": image_filename,
                    "size": len(image_bytes),
                    "dimensions": f"{width}x{height}" if width else "unknown"
                })
            except Exception as e:
                print(f"  ⚠️ 이미지 추출 실패 (페이지 {page_num + 1}): {e}")

    doc.close()
    return images_info


def extract_tables_camelot(pdf_path):
    """Camelot으로 표 정확하게 추출"""
    if not CAMELOT_AVAILABLE or not EXTRACT_TABLES:
        return []

    tables_data = []
    pdf_name = Path(pdf_path).stem
    Path(TABLE_DIR).mkdir(exist_ok=True)

    try:
        # lattice 모드: 선으로 구분된 표
        tables_lattice = camelot.read_pdf(str(pdf_path), pages='all', flavor='lattice')

        for i, table in enumerate(tables_lattice):
            # 정확도가 낮은 표는 제외
            if table.parsing_report['accuracy'] < 50:
                continue

            # CSV로 저장
            csv_filename = f"{pdf_name}_p{table.page}_table{i + 1}.csv"
            csv_path = Path(TABLE_DIR) / csv_filename
            table.to_csv(str(csv_path))

            # 마크다운 형식으로 변환
            markdown_table = table.df.to_markdown(index=False)

            tables_data.append({
                "page": table.page,
                "content": markdown_table,
                "csv_path": str(csv_path),
                "accuracy": table.parsing_report['accuracy'],
                "type": "lattice"
            })

        # stream 모드: 공백으로 구분된 표
        try:
            tables_stream = camelot.read_pdf(str(pdf_path), pages='all', flavor='stream')
            for i, table in enumerate(tables_stream):
                if table.parsing_report['accuracy'] < 50:
                    continue

                csv_filename = f"{pdf_name}_p{table.page}_table_stream{i + 1}.csv"
                csv_path = Path(TABLE_DIR) / csv_filename
                table.to_csv(str(csv_path))

                markdown_table = table.df.to_markdown(index=False)

                tables_data.append({
                    "page": table.page,
                    "content": markdown_table,
                    "csv_path": str(csv_path),
                    "accuracy": table.parsing_report['accuracy'],
                    "type": "stream"
                })
        except:
            pass  # stream 모드 실패는 무시

    except Exception as e:
        print(f"  ⚠️ Camelot 표 추출 실패: {e}")

    return tables_data


def extract_tables_basic(pdf_path):
    """기본 표 추출 (Camelot 없을 때)"""
    if not EXTRACT_TABLES:
        return []

    doc = fitz.open(pdf_path)
    tables_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # 표 형식 감지
        lines = text.split('\n')
        table_candidate = []

        for line in lines:
            # 탭 또는 다중 공백이 있고 숫자를 포함하는 경우
            if ('\t' in line or '  ' in line) and any(char.isdigit() for char in line):
                table_candidate.append(line)
            elif table_candidate and len(table_candidate) >= 3:
                # 3줄 이상이면 표로 판단
                tables_text.append({
                    "page": page_num + 1,
                    "content": '\n'.join(table_candidate),
                    "type": "basic"
                })
                table_candidate = []
            else:
                table_candidate = []

        # 마지막 테이블 처리
        if table_candidate and len(table_candidate) >= 3:
            tables_text.append({
                "page": page_num + 1,
                "content": '\n'.join(table_candidate),
                "type": "basic"
            })

    doc.close()
    return tables_text


def load_documents_with_media(docs_dir):
    """PDF/TXT 로드 + 이미지/표 추출"""
    paths = list(Path(docs_dir).rglob("*.pdf")) + list(Path(docs_dir).rglob("*.txt"))
    docs = []
    total_images = 0
    total_tables = 0

    for p in paths:
        try:
            if p.suffix.lower() == ".pdf":
                print(f"\n📄 처리 중: {p.name}")

                # 1. 이미지 추출
                images = extract_images_from_pdf(str(p))
                if images:
                    print(f"  📷 {len(images)} 개의 이미지 추출됨")
                    total_images += len(images)

                    for img_info in images:
                        img_doc = Document(
                            page_content=f"[이미지] {p.name} - 페이지 {img_info['page']}\n"
                                         f"파일명: {img_info['filename']}\n"
                                         f"크기: {img_info['size'] // 1024}KB, 해상도: {img_info['dimensions']}\n"
                                         f"경로: {img_info['path']}",
                            metadata={
                                "source": str(p),
                                "type": "image",
                                "page": img_info['page'],
                                "image_path": img_info['path'],
                                "filename": p.name
                            }
                        )
                        docs.append(img_doc)

                # 2. 표 추출 (Camelot 우선, 없으면 기본 모드)
                if CAMELOT_AVAILABLE:
                    tables = extract_tables_camelot(str(p))
                else:
                    tables = extract_tables_basic(str(p))

                if tables:
                    print(f"  📊 {len(tables)} 개의 표 추출됨")
                    total_tables += len(tables)

                    for table_info in tables:
                        table_content = f"[표] {p.name} - 페이지 {table_info['page']}\n\n{table_info['content']}"

                        metadata = {
                            "source": str(p),
                            "type": "table",
                            "page": table_info['page'],
                            "filename": p.name,
                            "extraction_method": table_info['type']
                        }

                        if 'csv_path' in table_info:
                            metadata['csv_path'] = table_info['csv_path']
                            metadata['accuracy'] = table_info['accuracy']

                        table_doc = Document(
                            page_content=table_content,
                            metadata=metadata
                        )
                        docs.append(table_doc)

                # 3. 원본 텍스트 추가
                pdf_docs = PyMuPDFLoader(str(p)).load()
                docs.extend(pdf_docs)
                print(f"  ✓ {len(pdf_docs)} 페이지 텍스트 로드됨")

            else:  # TXT 파일
                txt_docs = TextLoader(str(p), encoding="utf-8").load()
                docs.extend(txt_docs)
                print(f"✓ 로드됨: {p.name}")

        except Exception as e:
            print(f"✗ 실패: {p.name} - {e}")

    print(f"\n📊 추출 요약:")
    print(f"  - 총 이미지: {total_images}개")
    print(f"  - 총 표: {total_tables}개")
    print(f"  - 총 문서: {len(docs)}개")

    return docs


def main():
    print("=" * 60)
    print("🚀 PDF 인덱싱 시작 (고급 모드)")
    print("=" * 60)

    # 설정 출력
    print(f"\n⚙️ 설정:")
    print(f"  - 이미지 추출: {'✓' if EXTRACT_IMAGES else '✗'}")
    print(f"  - 표 추출: {'✓' if EXTRACT_TABLES else '✗'}")
    print(f"  - Camelot: {'✓ 사용' if CAMELOT_AVAILABLE else '✗ 기본 모드'}")
    print(f"  - 최소 이미지 크기: {MIN_IMAGE_SIZE // 1024}KB")

    # 문서 로드
    docs = load_documents_with_media(DOCS_DIR)

    if not docs:
        print("\n⚠️ 로드된 문서가 없습니다. docs/ 폴더를 확인하세요.")
        return

    # 텍스트 분할
    print(f"\n✂️ 텍스트 분할 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"  → {len(chunks)} 개의 청크 생성됨")

    # 임베딩 & 벡터DB 생성
    print(f"\n🔄 임베딩 생성 중... (시간이 걸릴 수 있습니다)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={'device': 'cpu'},  # GPU 있으면 'cuda'로 변경
        encode_kwargs={'normalize_embeddings': True}
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"\n✅ 인덱싱 완료!")
    print(f"  - 벡터 DB: {DB_DIR}")
    print(f"  - 이미지: {IMAGE_DIR}")
    if EXTRACT_TABLES and CAMELOT_AVAILABLE:
        print(f"  - 표 CSV: {TABLE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()