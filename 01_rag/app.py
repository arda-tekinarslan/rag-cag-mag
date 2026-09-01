"""
İspanyolca RAG Asistanı — Streamlit web arayüzü.
Çalıştırma (01_rag klasöründen):
    streamlit run app.py
"""

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from importlib import import_module
from sentence_transformers import SentenceTransformer, CrossEncoder
query_rewrite_mod = import_module("06_query_rewrite")

build_mode2 = import_module("02_build")
hybrid_mod = import_module("hybrid_rerank")
rag = import_module("03_generate_message")

st.set_page_config(page_title="İspanyolca RAG Asistanı", page_icon="🇪🇸", layout="centered")


@st.cache_resource(show_spinner="Sistem yükleniyor (ilk seferde birkaç saniye sürer)...")
def load_system():
    embeddings, chunks, metadata = build_mode2.build_index()
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    bm25 = hybrid_mod.build_bm25_index(chunks)
    cross_encoder = CrossEncoder(hybrid_mod.RERANK_MODEL, device="cpu")
    return embeddings, chunks, metadata, embed_model, bm25, cross_encoder


embeddings, chunks, metadata, embed_model, bm25, cross_encoder = load_system()

st.title("🇪🇸 İspanyolca Ders Materyali — RAG Asistanı")
st.caption(
    f"{len(chunks)} chunk · hybrid retrieval (BM25 + dense embedding) + cross-encoder rerank "
    f"+ {rag.MODEL_NAME}"
)

soru = st.text_input("Sorunuzu yazın:", placeholder="Örn: Reflexivo fiiller nasıl çekimlenir?")
use_rewriting = st.checkbox("Query rewriting kullan (daha yavaş, daha sağlam)")
sor_button = st.button("Sor", type="primary")

if sor_button and soru.strip():
    with st.spinner("Aranıyor ve cevap üretiliyor..."):
        if use_rewriting:
            retrieved_chunks, retrieved_meta, scores = query_rewrite_mod.retrieve_with_rewriting(
                soru, embeddings, chunks, metadata, embed_model, bm25, cross_encoder
            )
        else:
            retrieved_chunks, retrieved_meta, scores = rag.retrieve(
                soru, embeddings, chunks, metadata, embed_model, bm25, cross_encoder
            )
        prompt = rag.build_prompt(soru, retrieved_chunks)
        cevap = rag.generate(prompt)

    st.subheader("Cevap")
    st.write(cevap)

    with st.expander("Kullanılan kaynaklar (rerank skoruyla)"):
        for i, (meta, score, chunk) in enumerate(zip(retrieved_meta, scores, retrieved_chunks), 1):
            st.markdown(f"**[{i}]** `{score:.3f}` — {meta['topic']} / {meta['source']}")
            st.code(chunk[:400])

elif sor_button:
    st.warning("Lütfen bir soru yazın.")