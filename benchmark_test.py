import openvino_genai as ov_genai
import time
import os

model_name = "Qwen2.5-7B-Instruct-int4-ov"
model_path = os.path.abspath(f"../models/{model_name}")

print(f"--- Caricamento Jarvis sulla GPU Arc B50 ---")
pipe = ov_genai.LLMPipeline(model_path, "GPU")
print("✅ Sistema pronto. Analisi Benchmark attiva.\n")

while True:
    user_input = input("\n👤 Tu: ")
    if user_input.lower() in ["exit", "esci", "quit"]: break

    print("\n🤖 Jarvis: ", end="", flush=True)

    # Usiamo una lista per aggirare il problema dello scope
    stats = {"tokens": 0, "start_time": 0.0}

    def custom_streamer(word):
        # Se è il primo token, segnamo il tempo esatto di inizio generazione
        if stats["tokens"] == 0:
            stats["start_time"] = time.time()
        
        print(word, end="", flush=True)
        stats["tokens"] += 1
        return False 

    # Avviamo il timer totale (incluso il tempo di "pensiero" iniziale)
    overall_start = time.time()

    # Generazione
    pipe.generate(user_input, max_new_tokens=512, streamer=custom_streamer)
    
    overall_end = time.time()
    
    # Calcoli
    total_duration = overall_end - overall_start
    # Tempo di generazione pura (dal primo all'ultimo token)
    gen_duration = overall_end - stats["start_time"] if stats["start_time"] > 0 else 0
    
    tps = stats["tokens"] / gen_duration if gen_duration > 0 else 0

    print(f"\n\n" + "═"*40)
    print(f"📊 BENCHMARK B50:")
    print(f"⏱️  Tempo Totale: {total_duration:.2f} s")
    print(f"🚀 Velocità Pura: {tps:.2f} token/s")
    print(f"🔢 Token generati: {stats['tokens']}")
    print("═"*40)