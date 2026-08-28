"""
    Retrievel + LLM = RAG

"""

import ollama
import numpy as np
from importlib import import_module

build_mode = import_module("02_build")
MODEL_NAME = "qwen2.5:3b-instruct"

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

def retrieve(question:str,embeddings,chunks,metadata,model,k:int=3):
    q_vec = model.encode("query: " + question)
    scores = build_mode.cosine_sim(q_vec,embeddings)
    top_idx = build_mode.top_k_answ(scores,k)
    return [chunks[i] for i in top_idx],[metadata[i] for i in top_idx],[scores[i] for i in top_idx]


def generate(prompt:str)->str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role":"user","content":prompt}]
    )
    return response["message"]["content"]

def ask(question:str,embeddings,chunks,metadata,embed_model):
    retrieved_chunks,retrieved_metadata,scores = retrieve(question,embeddings,chunks,metadata,embed_model)

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")
    print("Resources:")

    for i,(metadata,score,chunk) in enumerate(zip(retrieved_metadata,scores,retrieved_chunks),1):
        print(f"  [{i}] ({score:.3f}) {metadata['topic']} / {metadata['source']}")
        print(f"      metin: {chunk[:200]}")

    prompt = build_prompt(question,retrieved_chunks)
    response = generate(prompt)

    print(f"\nRESPONSE:\n{response}")
    return response

def main():
    from sentence_transformers import SentenceTransformer

    embeddings,chunks,metadata = build_mode.build_index()
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")

    test_q = [
        "Reflexivo grammer kuralı nedir",
        "Juliete Venegas şarkısının ismi nedir",
        "Salatalık ispanyolcada ne demek"
        ]
    for q in test_q:
        ask(q,embeddings,chunks,metadata,embed_model)

if __name__ == "__main__":
    main()



