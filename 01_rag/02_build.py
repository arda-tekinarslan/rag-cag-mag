"""
    Chunk et textleri,embedd et,embedding matrix-metadata-metinleri diske kaydet,sorularla test
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from importlib import import_module

load_data = import_module("01_data_loading")
INDEX_PATH = Path(__file__).parent/"index.npz"

QUESTION_SET=[
    {"soru": "Reflexivo grammer kuralı nedir", "beklenen_konu": "4. Rutinas"},
    {"soru": "Vücudumuzdaki acıyı nasıl açıklarız", "beklenen_konu": "5.Estados fisicos y de animo"},
    {"soru": "İspanyolcada sabah rutini nasıl açıklanır", "beklenen_konu": "4. Rutinas"},
    {"soru": "Julieta Venegas'ın şarkısının ismi ne", "beklenen_konu": "3. Presente irregular"},
    {"soru": "İspanyolca spor isimleri nedir", "beklenen_konu": "5.Estados fisicos y de animo"},
    {"soru": "Salatalık İspanyolcada ne demek", "beklenen_konu": None}
]

def cosine_sim(query_vec:np.ndarray,doc_matrix:np.ndarray) ->np.ndarray: #(n,) and (d,n)
    scores = np.dot(doc_matrix,query_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_matrix,axis=1))
    return scores

def top_k_answ(scores:np.ndarray,k:int)->np.ndarray:
    top_k_indices = np.argsort(scores)[::-1][:k] #argsort artan sırada yapar o yüzden çevir
    return top_k_indices

def build_index() -> tuple[np.ndarray,list[str],list[dict]]:
    if INDEX_PATH.exists():
        print(f"Saved index found:{INDEX_PATH}")
        data = np.load(INDEX_PATH,allow_pickle=True) #Object array gibi kaydedilenleri yüklemeye izin verir
        return data["embeddings"],list(data["chunks"]),list(data["metadata"])
    
    print("No index found building index...")
    pdf_files = list(load_data.DATA_DIR.rglob("*.pdf"))
    docx_files = list(load_data.DATA_DIR.rglob("*.docx")) #rglob recursice method ile alt klasörlere girmeyi sağlar
    all_files = pdf_files + docx_files

    chunks,metadata = [],[]
    for file_path in all_files:
        topic = file_path.parent.name
        if file_path.suffix == ".pdf":
            text = load_data.extract_text_from_pdf(file_path)
        else:
            text = load_data.extract_text_from_docx(file_path)

        file_chunks = load_data.chunk_text(text)
        chunks.extend(file_chunks)
        metadata.extend({"topic":topic,"source":file_path.name} for _ in file_chunks)

    print(f"{len(chunks)} chunk is embedding this take some time...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    embeddings = model.encode(["passage: " + c for c in chunks],show_progress_bar=True)

    np.savez(
        INDEX_PATH,embeddings=embeddings,chunks=np.array(chunks,dtype=object),metadata=np.array(metadata,dtype=object)
    )
    print(f"Index kaydedildi: {INDEX_PATH}")
    return embeddings, chunks, metadata

def run_eval(embeddings,chunks,metadata,model):
    print("\n" + "=" * 60)
    print("EVAL")
    print("=" * 60)

    for item in QUESTION_SET:
        q = item["soru"]
        expected = item["beklenen_konu"]

        q_vec = model.encode("query: " + q)
        scores = cosine_sim(q_vec,embeddings)
        top_indices = top_k_answ(scores,3)

        topicsFound = [metadata[i]["topic"] for i in top_indices]
        hit = expected in topicsFound if expected else None 

        print(f"\nQuestion: {q}")
        print(f"  Expected sunject: {expected}")
        for rank,i in enumerate(top_indices,1):
            subject = metadata[i]["topic"]
            sign = " <-- Expected" if subject == expected else ""
            print(f"  {rank}. [{scores[i]:.3f}] ({subject}) {chunks[i][:80]}...{sign}")

        if expected is not None:
            print(f"  SONUÇ: {'✓ bulundu' if hit else '✗ bulunamadı'}")
        else:
            print(f"  SONUÇ: (negatif test — hiçbir konu 'doğru' değil, skorlara bak)")


def main():
    embeddings,chunks,metadata = build_index()
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    run_eval(embeddings,chunks,metadata,model)
    print("embeddings shape:", embeddings.shape)
    print("chunk sayısı:", len(chunks))
    print("metadata sayısı:", len(metadata))
    assert embeddings.shape[0] == len(chunks) == len(metadata)


if __name__ == "__main__":
    main()