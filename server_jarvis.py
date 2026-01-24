import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import openvino_genai as ov_genai
import openvino as ov
from huggingface_hub import snapshot_download
import uvicorn

model_name = "Qwen2.5-7B-Instruct-int4-ov"
model_path = f"../models/{model_name}"

if not os.path.exists(model_path) :
    print(f"\nScaricamento del modello {model_name} ottimizzato da Huggingface...")
    snapshot_download(f"OpenVINO/{model_name}", local_dir=model_path)
    print("\nDownload completato.")
else :
    print(f"\nModello {model_name} già presente localmente. Procedo al caricamento...")

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
    print(f"\nProvo a caricare il modello {model_name} sulla {model_device_name_GPU} da {model_path}")
    pipe = ov_genai.LLMPipeline(model_path, target_device)
    print(f"\nModello caricato correttamente su {model_device_name_GPU}")
except Exception as e :
    print(f"\nErrore caricamento su {model_device_name_GPU} : {e}")
    print(f"\nProvo a caricare il modello {model_name} su {model_device_name_CPU}...")
    pipe = ov_genai.LLMPipeline(model_path, "CPU")
    print(f"\nModello caricato correttamente su {model_device_name_CPU}")

app = FastAPI()

@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": "jarvis"}]
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    prompt = data["messages"][-1]["content"]
    
    def generate_stream() :
        def ov_streamer(subword: str) :
            chunk = {
                "choices": [{"delta": {"content": subword}, "index": 0, "finish_reason": None}]
            }
            return f"data: {json.dumps(chunk)}\n\n"
    for chunk in pipe.generate(prompt, max_new_tokens=512, streamer=ov_streamer):
            yield chunk
    yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)