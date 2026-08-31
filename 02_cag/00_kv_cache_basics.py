import copy
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" #Hugging face LLM modeli
CACHE_PATH = Path(__file__).parent / "toy-cache.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #Eğer GPU var NVDIA cuda onu kullan daha hızlı paralel işlem yapıyor büyük matrixler için hızlı

CONTEXT = """Sen bir İspanyolca dersi asistanısın. Aşağıdaki bilgileri bil:
- "Gato" kelimesi Türkçe'de "kedi" demektir.
- "Perro" kelimesi Türkçe'de "köpek" demektir.
- İspanyolca'da dönüşlü fiiller "se" eki alır, örnek: "lavarse" (kendini yıkamak).
- Julieta Venegas'ın en bilinen şarkısı "Limón y Sal"dır, 2006 yılında çıkmıştır.
"""
QUESTIONS = [
    "Gato ne demek?",
    "Julieta Venegas'ın şarkısı hangi yıl çıkmış?",
]

def build_prompt(context:str,question:str)->str:
    return f"{context}\nSoru:{question}\nCevap:"

#Cache hazır saklanan sabit bilgi hızlı ulaşım için onu oluşuturucaz.KV-Cache
def build_cache(model,tokenizer,context:str):
    context_inputs = tokenizer(context,return_tensors="pt").to(DEVICE) #Sonucu pytorch tensor olarak döndürüyor tokenize sonrası
    with torch.no_grad(): #Modeli eğitmediğimiz için gradient hesaplamaya gerek yok.Hız kazandırı böyle
        out = model(**context_inputs,use_cache=True) #KV leri sakla diyor use_cache ile
    return out.past_key_values,context_inputs["input_ids"].shape[1] #context token sayısını da return ediyoruz cachelenmiş contexti bilmek için 

def generate_with_cache(model,tokenizer,context:str,question:str,cache,max_new:int=60):
    prompt = build_prompt(context,question)
    question_inputs = tokenizer(prompt,return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(**question_inputs,past_key_values=cache,max_new_tokens=max_new,do_sample=False) #HuggingFace otomatik token üretme,en yüksek ihtimali seçer direk
        new_tokens = out_ids[0][question_inputs["input_ids"].shape[1]:] #burda out_ids tüm tokenlar var prompt dahil biz input_idsye olan kadar kısımdan kesip geri kalanın alyopruz
        return tokenizer.decode(new_tokens,skip_special_tokens=True)

def generate_without_cache(model,tokenizer,context:str,question:str,max_new:int=60):
    inputs = tokenizer(build_prompt(context,question),return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(**inputs,max_new_tokens=max_new,do_sample=False)
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens,skip_special_tokens=True)

def main():
    print(f"Cihaz: {DEVICE}")
    print(f"Model yükleniyor:{MODEL_NAME} (İLK SEFERDE 3GB İNDİRİR)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
    model.eval()

    print("\nCache oluşturuluyor (prefill)...\n")
    t0 = time.perf_counter()
    cache,context_length = build_cache(model,tokenizer,CONTEXT)
    prefill_s = time.perf_counter() - t0
    print(f"Context: {context_length} token | prefill: {prefill_s:.2f}s")

    torch.save(cache,CACHE_PATH) #KV Cache artık oluşturuldu ve kaydedildi
    print(f"Cache diske yazıldı: {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e6:.1f} MB)")

    for q in QUESTIONS:
        print(f"\n{'=' * 60}\nSORU: {q}")
        fresh = torch.load(CACHE_PATH,weights_only=False)#Her seferinde yeni cache yüklemesi ki eski sorununki kalmasın,weigthonly sadece tensor verilerini yüklemeye yarar

        t0 = time.perf_counter()
        answ_cached = generate_with_cache(model,tokenizer,CONTEXT,q,fresh)
        elapsed = time.perf_counter() - t0
        print(f"[CACHE'Lİ]  ({elapsed:.2f}s) {answ_cached.strip()}")

        t0 = time.perf_counter()
        answ_full = generate_without_cache(model,tokenizer,CONTEXT,q)
        elapsed = time.perf_counter() - t0
        print(f"[CACHE'SİZ] ({elapsed:.2f}s) {answ_full.strip()}")

if __name__ == "__main__":
    main()





    
