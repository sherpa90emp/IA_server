import os
import json
import threading
from queue import Queue
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import openvino_genai as ov_genai
import openvino as ov
from huggingface_hub import snapshot_download
import uvicorn

model_name = "Qwen3-1.7B-int4-ov"
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

model_lock = threading.Lock()

def 

def stream_generator(prompt, max_new_tokens, is_chat=False) :
    
    lock_acquired = model_lock.acquire(blocking=False)
    if not lock_acquired :
        yield f"data: {json.dumps({'error': 'GPU busy, blocked'})}\n\n"
        return
    
    try :
        token_queue = Queue()
        stop_event = threading.Event()

        def ov_streamer(subword: str) :
            if stop_event.is_set() :
                return True
            token_queue.put(subword)
            return False
    
        def run_generation() :
            try :
                pipe.generate(prompt, 
                              max_new_tokens=max_new_tokens, 
                              streamer=ov_streamer, 
                              do_sample=False,
                              num_beans=2, 
                              temperature=0.2, 
                              stop_tokens=["<|endoftext|>", "<|file_sep|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_prefix|>", "obj", "['middle_code']", "<|im_end|>", "<tool_call>", "<think>", "import "])
            except Exception as e :
                print(f"Errore generazione: {e}")
            finally : 
                token_queue.put(None)
    
        thread = threading.Thread(target=run_generation)
        thread.start()
    
        try :
            is_thinking = False
            while True :
                token = token_queue.get(timeout=5.0)
                if token is None :
                    break

                if "<think>" in token :
                    is_thinking = True
                    continue
                if "</think>" in token :
                    is_thinking = False
                    continue
                if is_thinking :
                    continue

                if "<tool_call>" in token or "</tool_call>" in token :
                    continue

                if token.strip() in ["```python", "```", "python", "<|fim_middle|>", "obj", "['middle_code']", "middle_code", "['", "']", "###"] :
                    continue
            
                if is_chat :
                    chunk = {
                        "choices": [{"delta": {"content": token}, "index": 0}] 
                    }
                else :
                    print(f"Inviando token: {token}")
                    chunk = {
                        "choices": [{"text": token, "index": 0}] 
                    }
                yield f"data: {json.dumps(chunk)}\n\n"
        except GeneratorExit :
            stop_event.set()
            print("Client disconnesso, segnale di stop inviato.")
            thread.join(timeout=1.0)
            raise
    finally :
        stop_event.set()
        thread.join(timeout=1.0)
        model_lock.release()  
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat(request: Request) :
    data = await request.json()
    prompt = data["messages"][-1]["content"]
    
    return StreamingResponse(stream_generator(prompt, 
                                              max_new_tokens=512, 
                                              is_chat=True), 
                                              media_type="text/event-stream")

@app.post("/v1/completions")
async def completions(request: Request) :
    data = await request.json()
    prompt = data.get("prompt", "")
    suffix = data.get("suffix", "")
    
    fim_prompt = {
        f"<|im_start|>system\nYou are a raw text completion tool. "
        f"DO NOT use <think>. DO NOT use <tool_call>. "
        f"Output ONLY the characters that follow the prefix.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    }

    return StreamingResponse(stream_generator(
        fim_prompt, 
        max_new_tokens=32,
        is_chat=False), 
        media_type="text/event-stream")

@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": "jarvis"}]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 