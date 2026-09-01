"""
    Hybrid retrievel DENSE + BM25 + CROSS-ENCODER
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer,CrossEncoder
from importlib import import_module

build_mode = import_module("02_build")
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

#
def build_bm25_index(chunks:list[str])->BM25Okapi:
    tokenized = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized) #Search için indekslendi chunklar

def bm25_search(question:str,bm25:BM25Okapi,k:int = 10)->list[int]:
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1][:k]
    return list(sorted_indices)

def reciprocal_rank_fusion(ranked_dense:list[int],ranked_bm25:list[int],k_constant:int=60)->list[int]:
    combined_scores = {}
    for rank,idx in enumerate(ranked_dense):
        combined_scores[idx] = combined_scores.get(idx,0) + (1 / (k_constant + rank))

    for rank,idx in enumerate(ranked_bm25):
        combined_scores[idx] = combined_scores.get(idx,0) + (1 / (k_constant + rank))

    sorted_combined = sorted(combined_scores,key=combined_scores.get,reverse=True)
    return sorted_combined

def hybrid_retrieve(question:str,embeddings,chunks,metadata,embed_model,bm25,n_candidates:int=10):
    q_vec =embed_model.encode("query: " + question)
    dense_score = build_mode.cosine_sim(q_vec,embeddings)
    ranked_dense = list(build_mode.top_k_answ(dense_score,n_candidates))

    ranked_bm25 = bm25_search(question,bm25,n_candidates)

    fused = reciprocal_rank_fusion(ranked_dense,ranked_bm25)
    return fused[:n_candidates]

def rerank(question:str,candidate_indices:list[int],chunks:list[str],cross_encoder:CrossEncoder,top_n:int=3): #iki ranked kısmından gelenleri en iyi şekilde sıralar cross encoder kullanraak
    pairs = [(question,chunks[i]) for i in candidate_indices]
    scores = cross_encoder.predict(pairs)

    order = np.argsort(scores)[::-1][:top_n]
    return [candidate_indices[i] for i in order],[scores[i] for i in order]

def compare(question:str,embeddings,chunks,metadata,embed_model,bm25,cross_encoder):
    print(f"\n{'='*70}")
    print(f"QUESTION: {question}")
    print(f"{'='*70}")

    q_vec = embed_model.encode("query: " + question)
    dense_score = build_mode.cosine_sim(q_vec,embeddings)
    dense_top3 = build_mode.top_k_answ(dense_score,3)

    print("\n[OLD] dense (embedding):")
    for rank, i in enumerate(dense_top3, 1):
        print(f"  {rank}. ({dense_score[i]:.3f}) {metadata[i]['topic']} | {chunks[i][:80]}...")
 
    candidates = hybrid_retrieve(question, embeddings, chunks, metadata, embed_model, bm25)
    final_idx, final_scores = rerank(question, candidates, chunks, cross_encoder, top_n=3)
 
    print("\n[YENİ] Hybrid (BM25+dense, RRF) + cross-encoder rerank:")
    for rank, (i, score) in enumerate(zip(final_idx, final_scores), 1):
        print(f"  {rank}. ({score:.3f}) {metadata[i]['topic']} | {chunks[i][:80]}...")

def main():
    embeddings, chunks, metadata = build_mode.build_index()
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
    cross_encoder = CrossEncoder(RERANK_MODEL)
    bm25 = build_bm25_index(chunks)
 
    test_sorular = [
        "Reflexivo grammer kuralı nedir",
        "Julieta Venegas'ın şarkısının ismi ne",
        "Salatalık İspanyolcada ne demek",  # negatif test
    ]
 
    for soru in test_sorular:
        compare(soru, embeddings, chunks, metadata, embed_model, bm25, cross_encoder)
 
 
if __name__ == "__main__":
    main()

