import os
import json
import threading
from queue import Queue, Empty
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import openvino_genai as ov_genai
import openvino as ov
from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVModelForCausalLM
from optimum.exporters.openvino.convert import export_tokenizer
from transformers import AutoTokenizer
import uvicorn

model_name = "Qwen/Qwen2.5-Coder-1.5B" 
if "-ov" in model_name :
    model_path = f"../models/{model_name.split('/')[-1]}"
else :
    model_path = f"../models/{model_name.split('/')[-1]}-ov"

if not os.path.exists(model_path) :
    if "OpenVINO" in model_name or "-ov" in model_name :
        print(f"\nScaricamento del modello {model_name} ottimizzato da Huggingface...")
        snapshot_download(model_name, local_dir=model_path)
        print("\nDownload completato.")
    else :
        print(f"\nModello OpenVINO non trovato. Avvio procedura di esportazione per {model_name}")
        print("Esportazione e quantizzazione int4 in corso (potrebbe richiedere qualche minuto)...")
        ov_model = OVModelForCausalLM.from_pretrained(
            model_name,
            export=True,
            compile=False,
            load_in_8bit=False,
            quantization_config={
                "bits": 4,
                "sym": True,
                "group_size": 128,
                "ratio": 0.8
            }    
        )
        ov_model.save_pretrained(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        export_tokenizer(tokenizer, model_path)
        print(f"Conversione completata. Modello salvato in: {model_path}")
        del ov_model
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
        )
    print(f"\nModello caricato correttamente su {model_device_name_GPU}")
    print("Tokenizer caricato correttamente")
except Exception as e :
    print(f"\nErrore caricamento su {model_device_name_GPU} : {e}")
    print(f"\nProvo a caricare il modello {model_name} su {model_device_name_CPU}...")
    pipe = ov_genai.LLMPipeline(model_path, "CPU")
    print(f"\nModello caricato correttamente su {model_device_name_CPU}")

app = FastAPI()

model_lock = threading.Lock()

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
                config = ov_genai.GenerationConfig()
                config.max_new_tokens = max_new_tokens

                if not is_chat :
                    config.do_sample = False
                    config.temperature = 0.0
                    config.presence_penalty = 1.5
                else :
                    config.do_samples = True
                    config.temperature = 0.6

                pipe.generate(prompt, config, streamer=ov_streamer)
            except Exception as e :
                print(f"Errore generazione: {e}")
            finally : 
                token_queue.put(None)
    
        thread = threading.Thread(target=run_generation)
        thread.start()
    
        try :
            while True :
                try :
                    token = token_queue.get(timeout=5.0)
                except Empty:
                    if stop_event.is_set() :
                        break
                    continue

                if token is None :
                    break

                if "<|" in token or "|>" in token or "Alibaba Cloud" in token:
                    continue

                if any(tag in token for tag in ["<tool_call>", "<think>", "</tool_call>", "<|im_end|>", "<|file_sep|>", "repo_name"]):
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
                                              is_chat=True,
                                              suffix=""), 
                                              media_type="text/event-stream")

@app.post("/v1/completions")
async def completions(request: Request) :
    data = await request.json()
    raw_prompt = data.get("prompt", "")
    print(f"Prompt modificato: {repr(raw_prompt)}")
    print("---------------------------------------")
    #full_prompt = raw_prompt.replace("\r\n<|fim_prefix>|", "<|fim_prefix|>")
    #full_prompt = full_prompt.replace("\n<|fim_suffix|>", "<|fim_suffix|>")
    #full_prompt = full_prompt.replace("<|fim_suffix|>\r\n", "<|fim_suffix|>")
    #full_prompt = full_prompt.replace("<|fim_middle|>\n", "<|fim_middle|>")
    
    full_prompt = full_prompt.replace(" <|fim_middle|>", "<|fim_middle|>")
    full_prompt = full_prompt.replace("\n<|fim_middle|>", "<|fim_middle|>")
    full_prompt = full_prompt.replace("\r\n<|fim_middle|>", "<|fim_middle|>")
    print(f"Prompt modificato: {repr(full_prompt)}")
    
    return StreamingResponse(stream_generator(
        full_prompt, 
        max_new_tokens=128,
        is_chat=False), 
        media_type="text/event-stream")

@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": "jarvis"}]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)