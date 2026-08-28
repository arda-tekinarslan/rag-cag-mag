import pymupdf  # PyMuPDF
import docx
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "spanish_pdf"


# ------------------------------------------------------------- TODO 1
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    pdf_path: bir .pdf dosyasının yolu
    dönüş:    dosyadaki tüm metin, tek bir string
    """
    doc = pymupdf.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text", sort=True))
    doc.close()
    return "\n\n".join(pages)

def extract_text_from_docx(docx_path:Path)->str:
    doc = docx.Document(docx_path)
    pages = []
    for parag in doc.paragraphs:
        text = parag.text
        pages.append(text)
    return "\n".join(pages)



# ------------------------------------------------------------- TODO 2
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 50) -> list[str]:
    """
    text:       tek parça uzun metin
    chunk_size: her parçanın karakter cinsinden hedef uzunluğu
    overlap:    ardışık parçalar arasında kaç karakter tekrar etsin

    dönüş: string listesi, her biri ~chunk_size karakterlik bir parça
    """
    chunks = []
    start = 0
    while start < len(text):
        chunk_i = text[start : start + chunk_size]
        chunks.append(chunk_i)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------- test
def _test():
    text = "A" * 1000  # 1000 karakterlik sahte metin
    chunks = chunk_text(text, chunk_size=300, overlap=50)
 
    assert all(len(c) <= 300 for c in chunks), "bir chunk 300 karakterden uzun"
    assert len(chunks) >= 4, f"beklenenden az chunk: {len(chunks)}"
    # overlap kontrolü: art arda iki chunk'ın kesişimi olmalı
    assert chunks[0][-50:] == chunks[1][:50], "overlap doğru çalışmıyor"
    print(f"✓ testler geçti — {len(chunks)} chunk üretildi\n")
 
 
# ---------------------------------------------------------------- main
def main():
    _test()
 
    if not DATA_DIR.exists():
        print(f"UYARI: {DATA_DIR} bulunamadı. PDF'lerini oraya koy.")
        return
 
    # rglob: alt klasörlere de iner (glob sadece üst seviyeye bakar)
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    docx_files = list(DATA_DIR.rglob("*.docx"))
    all_files = pdf_files + docx_files
    print(f"{len(pdf_files)} PDF, {len(docx_files)} docx bulundu.\n")
 
    all_chunks = []       # sadece metinler
    all_metadata = []     # her chunk için {"topic": ..., "source": ...}
    problematic = []      # (topic, source, karakter_sayısı) — boş veya çok kısa çıkanlar
 
    for file_path in all_files:
        topic = file_path.parent.name  # üst klasör adı = konu (örn. "7. Preterito perfecto")
 
        # uzantıya göre doğru extractor'ı seç
        if file_path.suffix == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_docx(file_path)
 
        chunks = chunk_text(text)
        print(f"[{topic}] {file_path.name}: {len(text)} karakter -> {len(chunks)} chunk")
 
        if len(text) < 100:
            problematic.append((topic, file_path.name, len(text)))
 
        all_chunks.extend(chunks)
        all_metadata.extend({"topic": topic, "source": file_path.name} for _ in chunks)
 
    print(f"\nToplam chunk sayısı: {len(all_chunks)}")
    print("\nİlk chunk örneği:")
    if all_chunks:
        print(f"  konu: {all_metadata[0]['topic']}")
        print(f"  metin: {all_chunks[0][:300]}")
 
    if problematic:
        print(f"\n--- {len(problematic)} PDF çok az/hiç metin verdi (muhtemelen taranmış/görsel) ---")
        for topic, name, n in problematic:
            print(f"  [{topic}] {name}: {n} karakter")
 
 
if __name__ == "__main__":
    main()