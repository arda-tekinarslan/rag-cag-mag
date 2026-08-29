"""
Temiz çıktı yardımcıları. 03_generate_message.py'nin en üstüne import et,
ask() fonksiyonunu aşağıdaki gibi güncelle.

Kullanım:
    from print_utils import print_question, print_sources, print_answer

    def ask(soru, embeddings, chunks, metadata, embed_model):
        retrieved_chunks, retrieved_meta, scores = retrieve(...)
        print_question(soru)
        print_sources(retrieved_meta, scores)
        cevap = generate(build_prompt(soru, retrieved_chunks))
        print_answer(cevap)
        return cevap
"""

WIDTH = 70


def print_question(soru: str):
    print("\n" + "┌" + "─" * (WIDTH - 2) + "┐")
    print(f"│ SORU: {soru}".ljust(WIDTH - 1) + "│")
    print("└" + "─" * (WIDTH - 2) + "┘")


def print_sources(retrieved_meta: list[dict], scores) -> None:
    print("  Kaynaklar:")
    for i, (meta, score) in enumerate(zip(retrieved_meta, scores), 1):
        print(f"    [{i}] ({score:.3f})  {meta['topic']} — {meta['source']}")


def print_answer(cevap: str):
    print("\n  CEVAP:")
    for line in cevap.strip().split("\n"):
        print(f"    {line}")
    print()