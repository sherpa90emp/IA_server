import openvino_genai as ov_genai
import openvino as ov
import time
import os

model_name = "Qwen2.5-7B-Instruct-int4-ov"
model_path = os.path.abspath(f"../models/{model_name}")

# Inizializzazione Pipeline
print(f"--- Caricamento Jarvis sulla GPU ---")
pipe = ov_genai.LLMPipeline(model_path, "GPU")
print("✅ Sistema pronto. Analisi Benchmark attiva.\n")

while True:
    user_input = input("\n👤 Tu: ")
    if user_input.lower() in ["exit", "esci", "quit"]: break

    print("\n🤖 Jarvis: ", end="", flush=True)

    # Variabili per il Benchmark
    tokens_count = 0
    start_time = time.time()

    # Funzione streamer modificata per contare i token
    def custom_streamer(word):
        nonlocal tokens_count
        print(word, end="", flush=True)
        tokens_count += 1
        return False # Continua la generazione

    # Generazione
    pipe.generate(user_input, max_new_tokens=512, streamer=custom_streamer)
    
    end_time = time.time()
    duration = end_time - start_time
    tps = tokens_count / duration if duration > 0 else 0

    # Stampa dei risultati tecnici
    print(f"\n\n" + "═"*40)
    print(f"📊 BENCHMARK B50:")
    print(f"⏱️  Tempo: {duration:.2f} sec")
    print(f"🚀 Velocità: {tps:.2f} token/s")
    print("═"*40)