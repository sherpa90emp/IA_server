import openvino_genai as ov_genai
import openvino as ov
from huggingface_hub import snapshot_download
import os

model_name = "Qwen2.5-7B-Instructint4-ov"
model_path = f"../models/{model_name}"

if not os.path.exists(model_path) :
    print(f"Scaricamento del modello {model_name} ottimizzato da Huggingface...")
    snapshot_download(f"OpenVINO/{model_name}", local_dir=model_path)
else :
    print(f"Modello {model_name} già presente localmente. Procedo al caricamento...")

core = ov.Core()
devices = core.available_devices

model_device_name_GPU = "Non trovata"
model_device_name_CPU = "Non trovata"
target_device = "CPU"

for device in devices : 
    full_name = core.get_property(device, "FULL_DEVICE_NAME")

    if "GPU" in full_name :
        model_device_name_GPU = full_name
        target_device = "GPU"
    elif "CPU" in full_name :
        model_device_name_CPU = full_name 

try :
    print(f"Provo a caricare il modello {model_name} sulla {model_device_name_GPU} da {model_path}")
    pipe = ov_genai.LLMPipeline(model_path, target_device)
    print(f"Modello caricato correttamente su {model_device_name_GPU}")
except Exception as e :
    print(f"Errore caricamento su {model_device_name_GPU} : {e}")
    print(f"Provo a caricare il modello {model_name} su {model_device_name_CPU}...")
    pipe = ov_genai.LLMPipeline(model_path, "CPU")
    print(f"Modello caricato correttamente su {model_device_name_CPU}")

while True :
    user_input = input("\nTu: ")
    if user_input.lower() in ["exit", "esci", "quit", "addio", "chiudi chat"] :
        break

    print("\nJarvis: ", end="", flush=True)
    pipe.generate(user_input, max_new_tokens=256, streamer=lambda word: print(word, end="", flush=True))
    print("\n")