import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

from importlib import import_module
from sentence_transformers import SentenceTransformer,CrossEncoder

rag = import_module("03_generate_message")
build_mode2 = import_module("02_build")
hybrid_mode = import_module("hybrid_rerank")

from print_utils import print_answer,print_sources,print_question
EXIT_WORDS = {"çıkış", "cikis", "exit", "q", "quit"}


def main():
    print("Sistem yükleniyor, bir saniye...")
    embeddings,chunks,metadata = build_mode2.build_index()
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
    bm25 = hybrid_mode.build_bm25_index(chunks)
    cross_encoder = CrossEncoder(hybrid_mode.RERANK_MODEL)

    print(f"\nHazır. {len(chunks)} chunk yüklendi. Soru sorabilirsin (çıkmak için 'çıkış' yaz).\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("Görüşürüz...")
            break

        rag.ask(question, embeddings, chunks, metadata, embed_model, bm25, cross_encoder)

if __name__ == "__main__":
    main() 
