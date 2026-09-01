from importlib import import_module
build_mode2 = import_module("02_build")
hybrid_mod = import_module("hybrid_rerank")
from sentence_transformers import SentenceTransformer

embeddings, chunks, metadata = build_mode2.build_index()
embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
bm25 = hybrid_mod.build_bm25_index(chunks)

soru = "Canción sobre limones a quién pertenece?"

# dense skoru
q_vec = embed_model.encode("query: " + soru)
dense_scores = build_mode2.cosine_sim(q_vec, embeddings)
print(f"Dense skor (chunk 9): {dense_scores[9]:.4f}")
print(f"Dense'de kaçıncı sırada: {sorted(dense_scores, reverse=True).index(dense_scores[9])}")

# bm25 skoru
tokenized_query = soru.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)
print(f"BM25 skor (chunk 9): {bm25_scores[9]:.4f}")
print(f"BM25'te kaçıncı sırada: {sorted(bm25_scores, reverse=True).index(bm25_scores[9])}")