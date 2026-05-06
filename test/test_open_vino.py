import openvino_genai as ov_genai
from huggingface_hub import snapshot_download
import os

# 1. Scarichiamo il modello già ottimizzato per OpenVINO da Intel
model_path = "Qwen2.5-1.5B-Instruct-openvino-int4"
if not os.path.exists(model_path):
    print("Scaricamento modello ottimizzato da HuggingFace...")
    snapshot_download("OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", local_dir=model_path)

# 2. Inizializziamo la Pipeline sulla tua B50
print(f"Caricamento modello sulla GPU B50...")
pipe = ov_genai.LLMPipeline(model_path, "GPU")

# 3. Generazione
print("\n--- TEST GENERAZIONE ---")
def streamer(subword):
    print(subword, end='', flush=True)
    return False

pipe.generate("Ciao! Sei un'IA che gira su Intel Arc B50. Dimmi tre cose belle di questa scheda.", 
              max_new_tokens=100, 
              streamer=streamer)
print("\n------------------------")