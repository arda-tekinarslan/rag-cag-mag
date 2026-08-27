"""
Hafta 0 — Retrieval'ın temeli: embedding + cosine similarity + top-k.
Kütüphane yok (FAISS/Chroma yok). Sadece numpy.

Kurulum:
    pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer #Metinleri sayılara dönüştürür

# ---------------------------------------------------------------- veri
CORPUS = [
    "FastAPI, Python için modern ve hızlı bir web framework'üdür.",
    "Pydantic, FastAPI'de veri doğrulama için kullanılır.",
    "Uvicorn, FastAPI uygulamalarını çalıştıran bir ASGI sunucusudur.",
    "Oracle veritabanında PL/SQL ile saklı yordam yazılabilir.",
    "SQL sorgularında JOIN, iki tabloyu ortak bir sütun üzerinden birleştirir.",
    "İndeksler veritabanı sorgularını hızlandırır ama yazma işlemini yavaşlatır.",
    "Faktoring, alacakların vadesinden önce nakde çevrilmesi işlemidir.",
    "Çek ve senet, faktoring işlemlerinde teminat olarak kullanılabilir.",
    "Kredi risk skorlaması, müşterinin temerrüde düşme olasılığını tahmin eder.",
    "Python'da list comprehension, döngüleri tek satırda yazmayı sağlar.",
    "Dekoratörler bir fonksiyonu sarmalayarak davranışını değiştirir.",
    "async/await, Python'da eşzamanlı G/Ç işlemleri için kullanılır.",
    "Git'te rebase, commit geçmişini doğrusal hale getirir.",
    "Docker imajı, uygulamayı bağımlılıklarıyla birlikte paketler.",
    "REST API'lerde GET isteği veri okumak, POST veri oluşturmak içindir.",
    "HTTP 401 kimlik doğrulama hatası, 403 yetki hatasıdır.",
    "JWT, istemci ile sunucu arasında güvenli token taşımak için kullanılır.",
    "Vektör veritabanları benzerlik aramasını hızlandırmak için kurulur.",
    "Embedding, metni anlamını koruyan sayısal bir vektöre dönüştürür.",
    "Cosine similarity, iki vektör arasındaki açının kosinüsünü ölçer.",
]

# ------------------------------------------------------------- TODO 1
def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    query_vec:  (d,)384      tek bir sorgu vektörü
    doc_matrix: (n, d)20,384    n adet doküman vektörü
    dönüş:      (n,)      her doküman için benzerlik skoru, [-1, 1] aralığında

    İpucu: cos(a,b) = (a·b) / (||a|| * ||b||)
    """
    sim = np.dot(doc_matrix,query_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_matrix,axis=1)) #Axis=1 demek sutünu yok et sonu.(20,) çıksın demek ve bize her satırın kendi normu lazım
    return sim


# ------------------------------------------------------------- TODO 2
def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """
    scores: (n,) benzerlik skorları
    dönüş:  (k,) en yüksek k skorun indeksleri, BÜYÜKTEN KÜÇÜĞE sıralı

    İpucu: np.argsort artan sıralar. Ters çevirmeyi unutma.
    """
    top_k_indices = np.argsort(scores)[::-1][:k]
    return top_k_indices


# ---------------------------------------------------------------- test
def _test():
    v = np.array([1.0, 0.0]) #(2,)
    m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 0.0]]) #(4,2)
    s = cosine_similarity(v, m)
    assert np.allclose(s, [1.0, 0.0, -1.0, 1.0]), f"cosine hatalı: {s}"
    assert s.shape == (4,), f"şekil hatalı: {s.shape}"
    assert list(top_k(np.array([0.1, 0.9, 0.5, 0.3]), 2)) == [1, 2], "top_k hatalı"
    print("✓ testler geçti\n")


# ---------------------------------------------------------------- main
def main():
    _test()

    model = SentenceTransformer("intfloat/multilingual-e5-small") #Hugging face hazır embedding modeli
    doc_matrix = model.encode(["passage: " + c for c in CORPUS]) #Liste verirsen (20,384),tek string(384,)

    sorular = [
        "Python web servisi nasıl yazılır?",
        "vadesi gelmemiş alacağı nakde çevirme",
        "iki metnin anlamca yakınlığını nasıl ölçerim?",
        "veritabanı sorgusu neden yavaş çalışır?",
    ]

    for soru in sorular:
        q = model.encode("query: " + soru) #Soruları embedding modeline göre vektörlere çevirirs
        scores = cosine_similarity(q, doc_matrix)
        print(f"SORU: {soru}")
        for rank, i in enumerate(top_k(scores, 3), 1):
            print(f"  {rank}. [{scores[i]:.3f}] {CORPUS[i]}")
        print()


if __name__ == "__main__":
    main()