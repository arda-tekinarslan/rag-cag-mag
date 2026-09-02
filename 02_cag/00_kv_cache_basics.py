import copy
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_PATH = Path(__file__).parent / "toy_cache.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #CUDA GPU çekirdeklerini kullanabilmemizi sağlar

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
    return f"{context}\nSoru: {question}\nCevap:"

def build_cache(model,tokenizer,context:str):
    context_inputs = tokenizer(context,return_tensors="pt").to(model.device)
    with torch.no_grad(): #Eğitim yok gradyan tutma
        out = model(**context_inputs,use_cache=True) #format geçişinde KV-cache üret ver
    return out.past_key_values,context_inputs["input_ids"].shape[1] # input_ids bize tokenize olmuş tensor tipindeki şeyi verir,outun past_key_valuesı CACHE dir

def generate_with_cache(model,tokenizer,context:str,question:str,cache,max_new:int=60) -> str:
    prompt = build_prompt(context,question)
    inputs = tokenizer(prompt,return_tensors="pt").to(model.device) #Promptu okumak için tokenize ediyoruz ve tensor nesnesi döndürüyoruz

    with torch.no_grad():
        out_ids = model.generate(**inputs,past_key_values=cache,max_new_tokens=max_new,do_sample=False) #cache ve elimdeki inputu kullanarak devamını üret,**inputs inputs dictini parametrelere ayırıyor inputs["input_ids"],inputs["attention_mask"] ...gibi,sampling yapma en yüksek olaslııklı tokenı seç direk
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:] #outids iki boyu[batchsize,sequencelength] 0 ile ,shape[1] olayı o uzunluktaki tokenlar önceden üretlienler ordan başla sona kadar yeniler zaten
    return tokenizer.decode(new_tokens,skip_special_tokens=True) #DECODE tokenize olmuş sayıları metne geri çeviriyor

def generate_without_cache(model,tokenizer,context,question,max_new:int=60)->str:
    prompt = build_prompt(context,question)
    inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs,max_new_tokens=max_new,do_sample=False)
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens,skip_special_tokens=True)

def main():
    print(f"Cihaz: {DEVICE}")
    print(f"Model yükleniyor: {MODEL_NAME} (ilk seferde ~3GB indirir)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
    model.eval()
    print("\nCache oluşturuluyor (prefill)...")
    t0 = time.perf_counter()
    cache,context_length = build_cache(model,tokenizer,CONTEXT)
    prefill_s = time.perf_counter() -t0
    print(f"Context: {context_length} token | prefill: {prefill_s:.2f}s")

    torch.save(cache,CACHE_PATH)
    print(f"Cache diske yazıldı: {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e6:.1f} MB)")

    for question in QUESTIONS:
        print(f"\n{'=' * 60}\nSORU: {question}")

        fresh = torch.load(CACHE_PATH,weights_only=False)
        t0 = time.perf_counter()
        response_cache = generate_with_cache(model,tokenizer,CONTEXT,question,fresh)
        t1 = time.perf_counter()

        print(f"[CACHE'Lİ]  ({t1 - t0:.2f}s) {response_cache.strip()}")
 
        t0 = time.perf_counter()
        cevap_full = generate_without_cache(model, tokenizer, CONTEXT, question)
        t1 = time.perf_counter()
        print(f"[CACHE'SİZ] ({t1 - t0:.2f}s) {cevap_full.strip()}")

if __name__ == "__main__":
    main()
