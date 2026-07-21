import re
import os
import json
import threading
from queue import Queue, Empty
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import openvino_genai as ov_genai
import openvino as ov
from transformers import AutoTokenizer
import uvicorn
from utilities.color_logger import ColoreLog
from utilities.general_func import rileva_device
from tools import TOOL_REGISTRY, execute_tool, get_schemas

class JarvisServerIDE:
    # Inizializza il server con modello e configurazioni base
    def __init__(self, model_name, model_path, model_type):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type

        self.app = FastAPI()
        self.model_lock = threading.Lock()
        self.pipe = None
        self.tokenizer = None

        self._setup_routes()

    # Carica il modello sul dispositivo hardware disponibile (priorità GPU)
    def load_hardware(self):
        """
        Carica il modello LLM sul dispositivo hardware disponibile (priorità GPU > CPU).
        1. Identifica i dispositivi OpenVINO disponibili (GPU Arc B50, CPU).
        2. Tenta di caricare il modello sulla GPU; in caso di fallimento, ricade sulla CPU.
        3. Utilizza il logger colorato per segnalare lo stato del caricamento.

        Nota: Il modello viene caricato tramite `LLMPipeline` con il percorso specificato,
        utilizzando la tokenizzazione da `AutoTokenizer` per il modello selezionato.
        """
        model_device_name_GPU, model_device_name_CPU, target_device = rileva_device()
                
        try :
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Provo a caricare il modello {self.model_name} sulla {model_device_name_GPU} da {self.model_path}")
            if self.model_type == "llm":
                self.pipe = ov_genai.LLMPipeline(self.model_path, target_device)
                
            else:
                self.pipe = ov_genai.VLMPipeline(self.model_path, target_device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                    )          
            print(f"\n{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Modello caricato correttamente su {model_device_name_GPU}")            
        except Exception as e :
            print(f"\n{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore caricamento su {model_device_name_GPU} : {e}")
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Provo a caricare il modello {self.model_name} su {model_device_name_CPU}...")
            self.pipe = ov_genai.LLMPipeline(self.model_path, "CPU")
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello caricato correttamente su {model_device_name_CPU}")     

    # Esegue generazione non-streaming del modello
    def _collect_generation(self, prompt: str, max_new_tokens: int, is_chat: bool) -> str:
        """
        Esegue la generazione e restituisce l'output completo come stringa.
        Non effettua streaming verso il client.
        Presuppone che il lock sia già acquisito dal chiamante.
 
        Args:
            prompt:         Prompt già formattato da apply_chat_template.
            max_new_tokens: Limite token da generare.
            is_chat:        True per chat (sampling), False per completions (greedy).
 
        Returns:
            Output grezzo completo del modello (incluso eventuale blocco <think>).
        """
        token_queue = Queue()
        stop_event = threading.Event()

        def ov_streamer(subword: str) -> bool:
            if stop_event.is_set() :
                return True
            token_queue.put(subword)
            return False
        
        def run_generation():
            try:
                config = ov_genai.GenerationConfig()
                config.max_new_tokens = max_new_tokens

                if not is_chat:
                    config.do_sample = False
                    config.temperature = 0.0
                    config.presence_penalty = 1.5
                else:
                    config.do_sample = True
                    config.temperature = 0.6

                self.pipe.generate(prompt, config, streamer=ov_streamer)
            except Exception as e:
                print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore generazione: {e}")
            finally:
                token_queue.put(None)
        
        thread = threading.Thread(target=run_generation)
        thread.start()

        output = ""
        while True:
            try:
                token = token_queue.get(timeout=5.0)
            except Empty:
                continue
            if token is None:
                break
            output += token

        thread.join()

        return output
    
    # Genera output in streaming con controllo token
    def stream_generator(self, prompt: str, max_new_tokens: int, is_chat=False, **kwargs):
        """
        Genera l'output del modello in streaming, con controllo sui token e gestione degli stop.
        Filtra i token non desiderati e restituisce i frammenti di risposta via SSE.

        Args:
            prompt:         Prompt già formattato per l'input del modello.
            max_new_tokens: Limite massimo di token da generare.
            is_chat:        True per modalità chat (sampling), False per completamento (greedy).
            **kwargs:       Parametri aggiuntivi (es. max_new_tokens, presence_penalty).

        Yields:
            Frammenti di risposta in formato SSE (Server-Sent Events).
        """
        max_new_tokens = kwargs.get("max_new_tokens", max_new_tokens)
        print(f"{ColoreLog.DEBUG}[STREAM]{ColoreLog.RESET} max_new_tokens={max_new_tokens} | kwargs keys={list(kwargs.keys())}")

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
                        config.do_sample = True
                        config.temperature = 0.6

                    self.pipe.generate(prompt, config, streamer=ov_streamer)
                except Exception as e :
                    print(f"Errore generazione: {e}")
                finally : 
                    token_queue.put(None)
        
            thread = threading.Thread(target=run_generation)
            thread.start()

            try :
                found_and_think = False
                think_buffer = ""

                while True :
                    try :
                        token = token_queue.get(timeout=5.0)
                    except Empty:
                        if stop_event.is_set() :
                            break
                        continue

                    if token is None :
                        break

                    if not is_chat :
                        if any(s in token for s in ["README.md", "Copyright", "---", "repo_name", "/*", "*/"]):
                            print(f"--- STOP: Rilevato tentativo di cambiare file ({token.strip()}) ---")
                            stop_event.set()
                            break

                    if any(tag in token for tag in ["<|", "|>", "Alibaba Cloud", "AlibabaCloud"]):
                        continue

                    if not found_and_think:
                        think_buffer += token
                        print(f"{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Token ricevuto: {repr(token)} | in_think: {found_and_think} | buffer tail: {repr(think_buffer)}")
                        if "</think>" in think_buffer:
                            found_and_think = True
                            after_think = think_buffer.split("</think>", 1)[-1]
                            think_buffer = ""
                            if not after_think.strip():
                                continue
                            token = after_think
                        else:
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
            except GeneratorExit:
                stop_event.set()
                print("Client disconnesso, segnale di stop inviato.")
                thread.join(timeout=1.0)
                raise
        finally :
            stop_event.set()
            thread.join(timeout=1.0)
            self.model_lock.release()  
            yield "data: [DONE]\n\n"

    # Gestisce il ciclo di chiamate tool e streaming finale
    def tool_stream_generator(self, message: list, prompt: str, max_new_tokens: int):
        """
        Gestisce il ciclo completo di tool calling:
        1. Genera l'output completo (senza streaming).
        2. Se contiene <tool_call>: esegue il tool, reinserisce il risultato e rigenera.
        3. Alla fine, invia la risposta pulita in streaming al client.
 
        Supporta fino a 3 chiamate tool consecutive prima di forzare la risposta finale.
 
        Args:
            messages:       Lista messaggi OpenAI-format (per ricostruire il prompt).
            prompt:         Prompt iniziale già formattato con gli schemi tool.
            max_new_tokens: Limite token per ogni generazione.
        """
        lock_acquired = self.model_lock.acquire(blocking=False)
        if not lock_acquired:
            yield f"data: {json.dumps({'error': 'GPU busy, blocked'})}\n\n"
            return

        try:
            current_message = list(message)
            current_prompt = prompt
            MAX_TOOL_CALLS = 3

            for attempt in range (MAX_TOOL_CALLS + 1):
                print(f"{ColoreLog.INFO}[TOOL_STREAM]{ColoreLog.RESET} Tentativo {attempt + 1}/{MAX_TOOL_CALLS + 1} — generazione in corso...")
                raw_output = self._collect_generation(current_prompt, max_new_tokens, is_chat=True)
                print(f"{ColoreLog.DEBUG}[TOOL_STREAM]{ColoreLog.RESET} Raw output ({len(raw_output)} chars): {repr(raw_output[:200])}")

                clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
                print(f"{ColoreLog.DEBUG}[TOOL_STREAM]{ColoreLog.RESET} Clean output dopo rimozione <think>: {repr(clean_output[:200])}")

                tool_match = re.search(r"<tool_call>(.*?)</tool_call>", clean_output, flags=re.DOTALL)

                if tool_match and attempt < MAX_TOOL_CALLS:
                    tool_json_str = tool_match.group(1).strip()
                    print(f"{ColoreLog.INFO}[TOOL_STREAM]{ColoreLog.RESET} Tool call rilevata: {repr(tool_json_str[:200])}")

                    try:
                        tool_data = json.loads(tool_json_str)
                        name = tool_data.get("name")
                        arguments = tool_data.get("arguments", {})

                        print(f"{ColoreLog.INFO}[TOOL]{ColoreLog.RESET} Chiamata {name} ({arguments})")
                        result = execute_tool(name, arguments)
                        print(f"{ColoreLog.SUCCESS}[TOOL]{ColoreLog.RESET} Risultato: {result[:120]}")

                        current_message.append({"role": "assistant", "content": raw_output})
                        current_message.append({"role": "tool", "content": result, "name": name})

                        current_prompt = self.tokenizer.apply_chat_template(
                            current_message,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        continue
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore parsing tool call: {e}")
                
                final_text = re.sub(r"<\|[^|]*\|+>", "", clean_output).strip()
                final_text = re.sub(r"<tool_call>.*?</tool_call>", "", final_text, flags=re.DOTALL).strip()
                final_text = re.sub(r"<tool_response>.*?</tool_response>", "", final_text, flags=re.DOTALL).strip()

                print(f"{ColoreLog.SUCCESS}[TOOL_STREAM]{ColoreLog.RESET} Risposta finale ({len(final_text)} chars): {repr(final_text[:200])}")

                CHUNK_SIZE = 20
                for i in range(0, len(final_text), CHUNK_SIZE):
                    chunk_text = final_text[i:i + CHUNK_SIZE]
                    chunk = {"choices": [{"delta": {"content": chunk_text}, "index": 0}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                break
        finally:
            self.model_lock.release()
            yield "data: [DONE]\n\n"    

    # Configura le route API per le API OpenAI-compatibili
    def _setup_routes(self):
        """
        Configura le route FastAPI per le API OpenAI-compatibili:
        - POST /v1/chat/completions: Gestione chat con supporto strumenti.
        - POST /v1/completions: Gestione completamenti (autocompletamento).
        - GET /v1/models: Lista dei modelli disponibili.
        """
        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            data = await request.json()
            messages = data.get("messages", [])
            
            client_wants_tools = data.get("tools") or data.get("tool_choice")
            schemas = get_schemas() if client_wants_tools else None

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=schemas if schemas else None,
                tokenize=False,
                add_generation_prompt=True
            )
            
            if schemas:
                return StreamingResponse(
                    self.tool_stream_generator(
                        messages,
                        prompt,
                        max_new_tokens=4096
                    ),
                    media_type="text/event-stream"
                )
            else:
                return StreamingResponse(
                    self.stream_generator(
                        prompt, 
                        max_new_tokens=4096, 
                        is_chat=True,
                        **data
                        ), 
                        media_type="text/event-stream")

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            data = await request.json()
            raw_prompt = data.get("prompt", "")
            
            #full_prompt = re.sub(r'[\s\r\n]+<\|fim_middle\|>', '<|fim_middle|>', raw_prompt)

            print(f"Prompt originale: {repr(raw_prompt)}")
            #print("---------------------------------------")
            #print(f"Prompt modificato: {repr(full_prompt)}")
            
            return StreamingResponse(self.stream_generator(
                raw_prompt,
                #full_prompt, 
                max_new_tokens=64, #tenere token bassi e debouncing a 100ms sembra possa facilitare l'autocompletamento
                is_chat=False), 
                media_type="text/event-stream")

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "data": [{"id": "jarvis"}]
            }

    # Avvia il server FastAPI con l'API OpenAI-compatibile    
    def run_server_IDE(self, host="0.0.0.0", port=8000):
        """
        Avvia il server FastAPI con l'API OpenAI-compatibile.

        Args:
            host: Indirizzo IP del server (default: "0.0.0.0").
            port: Numero della porta (default: 8000).
        """
        self.load_hardware()
        print(f"\n[READY] Server Jarvis attivo su https://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)