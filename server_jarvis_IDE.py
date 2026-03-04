import re
import json
import threading
from queue import Queue, Empty
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import openvino_genai as ov_genai
import openvino as ov
from transformers import AutoTokenizer
import uvicorn

class JarvisServeIDE:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.app = FastAPI()
        self.model_lock = threading.Lock()
        self.pipe = None
        self.tokenizer = None

        self._setup_routes()

    def load_hardware(self):
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
            print(f"\n[INFO] Provo a caricare il modello {self.model_name} sulla {model_device_name_GPU} da {self.model_path}")
            self.pipe = ov_genai.LLMPipeline(self.model_path, target_device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
                )
            print(f"\n[SUCCESS] Modello caricato correttamente su {model_device_name_GPU}")
            print("[SUCCESS] Tokenizer caricato correttamente")
        except Exception as e :
            print(f"\n[ERROR] Errore caricamento su {model_device_name_GPU} : {e}")
            print(f"\n[INFO] Provo a caricare il modello {self.model_name} su {model_device_name_CPU}...")
            self.pipe = ov_genai.LLMPipeline(self.model_path, "CPU")
            print(f"\nModello caricato correttamente su {model_device_name_CPU}")     

    def stream_generator(self, prompt, max_new_tokens, is_chat=False):
        lock_acquired = self.model_lock.acquire(blocking=False)
        if not lock_acquired:
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

                    self.pipe.generate(prompt, config, streamer=ov_streamer)
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

                    if any(s in token for s in ["README.md", "Copyright", "---", "repo_name", "/*", "*/"]):
                        print(f"--- STOP: Rilevato tentativo di cambiare file ({token.strip()}) ---")
                        stop_event.set()
                        break

                    if any(tag in token for tag in ["<|", "|>", "Alibaba Cloud", "<tool_call>", "<think>", "AlibabaCloud"]):
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
            self.model_lock.release()  
            yield "data: [DONE]\n\n"

    def _setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat(request: Request) :
            data = await request.json()
            prompt = data["messages"][-1]["content"]
            
            return StreamingResponse(self.stream_generator(prompt, 
                                                    max_new_tokens=512, 
                                                    is_chat=True,
                                                    suffix=""), 
                                                    media_type="text/event-stream")

        @self.app.post("/v1/completions")
        async def completions(request: Request) :
            data = await request.json()
            raw_prompt = data.get("prompt", "")
            
            full_prompt = re.sub(r'[\s\r\n]+<\|fim_middle\|>', '<|fim_middle|>', raw_prompt)

            #print(f"Prompt originale: {repr(raw_prompt)}")
            #print("---------------------------------------")
            #print(f"Prompt modificato: {repr(full_prompt)}")
            
            return StreamingResponse(self.stream_generator(
                full_prompt, 
                max_new_tokens=24, #tenere token bassi e debouncing a 100ms sembra possa facilitare l'autocompletamento
                is_chat=False), 
                media_type="text/event-stream")

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "data": [{"id": "jarvis"}]
            }  

    def run_server_IDE(self, host="0.0.0.0", port=8000):
        self.load_hardware()
        print(f"\n[READY] Server Jarvis attivo su https://{host}:{port}")
        uvicorn.run(app, host=host, port=port)