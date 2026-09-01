"""
    Kullanıcının sorusuna direk güvenmek yerine LLM e birkaç farklı versionda soru veriyor
    Her versiyona hybrid retrievel yapıcaz ve RRF ile birleştiricez
"""

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1" #Ekstra bilgi gönderimini iptal eder güvenlik için önemli
from importlib import import_module
from sentence_transformers import SentenceTransformer,CrossEncoder

build_mode_2 = import_module("02_build")
hybrid_mode = import_module("hybrid_rerank")
rag = import_module("03_generate_message")

def build_rewrite_prompt(question:str)->str:
    role = f"Sen verdiğim örnekleri analiz ederek yeni promptlar üreten bir uzmansın"

    rule = (
    "Sana verdiğim örneklerdeki gibi üret. "
    "SADECE 3 satır döndür — başka hiçbir açıklama, giriş cümlesi, "
    "numara veya madde işareti EKLEME. "
    "1. satır MUTLAKA İspanyolca bir çeviri olmalı (çünkü kaynak dökümanlar İspanyolca). "
    "2. ve 3. satırlar Türkçe ama farklı kelimelerle ifade edilmiş olsun."
)

    example = f"""
        Soru: "Kediler ne yer?"
        Cevap:
        ¿Qué comen los gatos?
        Kediler neyi yiyerek besleniyor?
        Beslenme kaynağı olarak kediler hangi gıdaları tüketir?
    """

    prompt = f"""
    ROLE:{role},
    RULE:{rule},
    EXAMPLE:{example},
    QUESTION:{question}
    """
    return prompt

def parse_variants(llm_response:str,k:int=3) -> list[str]:
    lines = llm_response.strip().split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        if(line):
            clean_lines.append(line.lstrip("0123456789.-) ").strip())
    return clean_lines[:k]

def reciprocal_rank_fusion_multi(ranked_list:list[list[int]],k_constant:int=60)->list[int]: #Her chunkun her queryiden aldığı puanları bulucaz sonra chunklar için tek bir liste yapıcaz
    combined_scores = {}
    for ranked in ranked_list:
        for rank,idx in enumerate(ranked):
            combined_scores[idx] = combined_scores.get(idx,0) + (1 / (k_constant + rank)) #O indeksteki puan + bizim skor

    return sorted(combined_scores,key=combined_scores.get,reverse=True) #Ranklara göre büyükten küçüğe sıralama


def retrieve_with_rewriting(question:str,embeddings,chunks,metadata,embed_model,bm25,cross_encoder,k=3):
    rewrite_prompt = build_rewrite_prompt(question)
    llm_response = rag.generate(rewrite_prompt) #Prompt sonucu çıktımız
    variants = parse_variants(llm_response)

    print(f"  [query rewriting] Üretilen varyasyonlar: {variants}")

    all_query = [question] + variants
    all_ranked = []
    for q in all_query:
        candidates = hybrid_mode.hybrid_retrieve(q,embeddings,chunks,metadata,embed_model,bm25,n_candidates=10)
        all_ranked.append(candidates)
    fused = reciprocal_rank_fusion_multi(all_ranked)
    unique_candidates = fused[:15]
    final_idx,final_score = hybrid_mode.rerank(question,unique_candidates,chunks,cross_encoder,top_n=k)

    return [chunks[i] for i in final_idx],[metadata[i] for i in final_idx],final_score

def ask_with_rewriting(q,embeddings,chunks,metadata,embed_model,bm25,cross_encoder):
    from print_utils import print_answer,print_question,print_sources
    print_question(q)
    retrieved_chunks,retrieved_metadata,scores = retrieve_with_rewriting(q,embeddings,chunks,metadata,embed_model,bm25,cross_encoder)
    print_sources(retrieved_metadata,scores)
    prompt = rag.build_prompt(q,retrieved_chunks)
    response = rag.generate(prompt)
    print_answer(response)
    return response

def main():
    embeddings,chunks,metadata = build_mode_2.build_index()
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small",device="cpu")
    bm25 = hybrid_mode.build_bm25_index(chunks)
    cross_encoder = CrossEncoder(hybrid_mode.RERANK_MODEL,device="cpu")

    test_sorular = [
        "Limón y Sal kimin şarkısı",
        "Reflexivo grammer kuralı nedir",
        "Salatalık İspanyolcada ne demek",  # negatif test, hâlâ "yok" demeli
    ]

    for soru in test_sorular:
        ask_with_rewriting(soru, embeddings, chunks, metadata, embed_model, bm25, cross_encoder)

if __name__ == "__main__":
    main()

