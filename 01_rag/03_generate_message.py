"""
    Retrievel + LLM = RAG

"""

import ollama
from sentence_transformers import CrossEncoder
import numpy as np
from importlib import import_module
from print_utils import print_question,print_sources,print_answer
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

build_mode = import_module("hybrid_rerank")
build_mode2 = import_module("02_build")
MODEL_NAME = "qwen2.5:7b-instruct-q3_K_M"

def build_prompt(question:str,retrieved_chunks:list[str])->str: #Retrieveed chunks top_k den gelen chunklar scorea göre
    numbered_chunks = []
    for i,chunk in enumerate(retrieved_chunks,1):
        numbered_chunks.append(f"[{i}]" + chunk)
    numbered_contex = "\n\n".join(numbered_chunks)

    role = "Sen İspanyolca dilinde uzman bir ai öğretmenisin."
    rule = (
    "SADECE aşağıdaki CONTEXT içindeki bilgileri kullanarak cevap ver. "
    "Kendi genel bilgini veya context dışındaki hiçbir bilgiyi KULLANMA. "
    "Eğer cevap context içinde yoksa, başka hiçbir şey eklemeden sadece şunu yaz: "
    "'Bu bilgi elimdeki dokümanlarda yok.' "
    "Cevabını hangi chunk numarasından (örn. [1], [2]) aldığını belirt."
)

    prompt = f"""
        ROLE:{role}
        RULE:{rule}
        CONTEXT:{numbered_contex}
        QUESTION:{question}
        """
    return prompt

def retrieve(question: str, embeddings, chunks, metadata, model, bm25, cross_encoder, k: int = 3):
    candidates = build_mode.hybrid_retrieve(question, embeddings, chunks, metadata, model, bm25, n_candidates=10)
    final_idx, final_scores = build_mode.rerank(question, candidates, chunks, cross_encoder, top_n=k)
    return [chunks[i] for i in final_idx], [metadata[i] for i in final_idx], final_scores

def generate(prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return response["message"]["content"]

def ask(question, embeddings, chunks, metadata, embed_model, bm25, cross_encoder):
    retrieved_chunks, retrieved_meta, scores = retrieve(question, embeddings, chunks, metadata, embed_model, bm25, cross_encoder)

    print_question(question)
    print_sources(retrieved_meta, scores)
    print("\n--- TAM CHUNK METNİ (debug) ---")
    print(retrieved_chunks[0])
    print("--- son ---\n")

    prompt = build_prompt(question, retrieved_chunks)
    cevap = generate(prompt)

    print_answer(cevap)
    return cevap

def main():
    from sentence_transformers import SentenceTransformer

    embeddings, chunks, metadata = build_mode2.build_index()
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
    bm25 = build_mode.build_bm25_index(chunks)
    cross_encoder = CrossEncoder(build_mode.RERANK_MODEL)

    test_sorular = [
    "Reflexivo grammer kuralı nedir",
    "Julieta Venegas'ın şarkısının ismi ne",
    "Salatalık İspanyolcada ne demek",
]
    for soru in test_sorular:
        ask(soru, embeddings, chunks, metadata, embed_model, bm25, cross_encoder)

if __name__ == "__main__":
    main()



