import openvino_genai as ov_genai
import os

# Ottieni il percorso assoluto della cartella scaricata
model_path = os.path.abspath("../models/Qwen1.5B-OV")

print(f"Sto caricando Jarvis dalla cartella: {model_path}")
print("Caricamento sulla GPU Intel Arc B50 in corso...")

# Inizializziamo la pipeline sulla GPU
try:
    pipe = ov_genai.LLMPipeline(model_path, "GPU")
    print("✅ Jarvis è ONLINE sulla GPU!")
except Exception as e:
    print(f"❌ Errore nel caricamento sulla GPU: {e}")
    print("Provo sulla CPU...")
    pipe = ov_genai.LLMPipeline(model_path, "CPU")

# Chat loop
while True:
    user_input = input("\nTu: ")
    if user_input.lower() in ["exit", "esci", "quit"]:
        break
    
    print("\nJarvis: ", end="", flush=True)
    pipe.generate(user_input, max_new_tokens=256, streamer=lambda word: print(word, end="", flush=True))
    print("\n")