import re
import time
import json
import threading
from queue import Queue, Empty
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import openvino_genai as ov_genai
import openvino as ov
from transformers import AutoTokenizer
import uvicorn
import uuid
from utilities.color_logger import ColoreLog
from utilities.general_func import rileva_device
from tools import TOOL_REGISTRY, execute_tool, get_schemas

class JarvisServerIDE:
    # Inizializza il server con modello e configurazioni base
    def __init__(self, model_name, model_path, model_type, model_draft, model_draft_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type

        self.app = FastAPI()
        self.model_lock = threading.Lock()
        self.pipe = None
        self.tokenizer = None
        self.draft_pipe = None

        # --- KV-Cache configurazione ---
        self.kv_cache_quantization = "f16"
        self.cache_eviction_enabled = True

        # --- Speculative Decoding / Tree Search ---
        self.model_draft_name = model_draft
        self.model_draft_path = model_draft_path
        self.num_assistant_tokens = 5       # token candidati totali inviati al target model
        self.branching_factor = 2           # candidati per nodo dell'albero di ricerca
        self.tree_depth = 2                 # profondità lookahead del draft model

        self._setup_routes()

    # Carica il modello sul dispositivo hardware disponibile (priorità GPU)
    def load_hardware(self):
        """
        Carica il modello VLM o LLM sul dispositivo hardware disponibile (priorità GPU > CPU).
        1. Identifica i dispositivi OpenVINO disponibili (GPU Arc B50, CPU).
        2. Tenta di caricare il modello sulla GPU; in caso di fallimento, ricade sulla CPU.
        3. Utilizza il logger colorato per segnalare lo stato del caricamento.

        Nota: Il modello viene caricato tramite `LLMPipeline` con il percorso specificato,
        utilizzando la tokenizzazione da `AutoTokenizer` per il modello selezionato.
        """

        model_device_name_GPU, model_device_name_CPU, gpu_device_id = rileva_device()
     
        try :
            print(f"{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET} Desideri utilizzare il metodo di caricamento HETERO? s/n Premendo INVIO si utilizzarà il metodo HETERO")
            user_input_caricamento_gpu = input()

            if user_input_caricamento_gpu.lower() == "s" or not user_input_caricamento_gpu :
                target_device = "HETERO:" + ",".join(gpu_device_id)
            elif user_input_caricamento_gpu.lower() == "n":
                target_device = "GPU"
            else:
                print(f"")

            pipeline_kwargs = {}
            pipeline_kwargs["KV_CACHE_PRECISION"] = self.kv_cache_quantization
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} KV_CACHE_PRECISION: {self.kv_cache_quantization}")

            scheduler_config = None

            if self.cache_eviction_enabled:
                scheduler_config = scheduler_config or ov_genai.SchedulerConfig()
                scheduler_config.use_cache_eviction = True
                scheduler_config.cache_eviction_config = ov_genai.CacheEvictionConfig(
                    256,                                                                    # start_size
                    512,                                                                    # recent_size
                    4096,                                                                   # max_cache_size
                    aggregation_mode=ov_genai.AggregationMode.NORM_SUM,
                    apply_rotation=True,
                    snapkv_window_size=8,
                )
                print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} EVICTION: {self.cache_eviction_enabled}")

            draft_model = None

            if self.model_draft_path and target_device.startswith("HETERO"):
                draft_device = gpu_device_id[0]
                draft_model = ov_genai.draft_model(self.model_draft_path, draft_device)
                scheduler_config = scheduler_config or ov_genai.SchedulerConfig()
                scheduler_config.cache_size=4
                scheduler_config.max_num_seqs=4
                scheduler_config.dynamic_split_fuse=True
                scheduler_config.enable_prefix_caching=True

                pipeline_kwargs["DRAFT_MODEL"] = draft_model
                
                self.draft_pipe = draft_model

                print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Speculative decoding abilitato: modello {self.model_draft_name} sulla {draft_device} da {self.model_draft_path}\n")
                print(f"{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Configurazione speculative decoding: {scheduler_config.to_string()}\n")

            if scheduler_config is not None:
                pipeline_kwargs["SCHEDULER_CONFIG"] = scheduler_config

            elif self.model_draft_path and target_device == "GPU":
                print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Speculative decoding non abilitato.\n")
                self.draft_pipe = None

            else:
                print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Speculative decoding non attivo.\n")
                self.draft_pipe = None

            if self.model_type == "llm":
                if target_device == "GPU":
                    print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Provo a caricare il modello {self.model_name} di tipo {self.model_type} sulla {model_device_name_GPU[0]} da {self.model_path}")
                    self.pipe = ov_genai.LLMPipeline(
                        self.model_path, 
                        target_device,
                        **pipeline_kwargs
                        )
                else:
                    print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Provo a caricare il modello {self.model_name} di tipo {self.model_type} su entrambe le {model_device_name_GPU[0]} da {self.model_path}")
                    self.pipe = ov_genai.LLMPipeline(
                        self.model_path, 
                        target_device, 
                        MODEL_DISTRIBUTION_POLICY="PIPELINE_PARALLEL",
                        **pipeline_kwargs
                        )
            else:
                if target_device == "GPU":
                    print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Provo a caricare il modello {self.model_name} di tipo {self.model_type} sulla {model_device_name_GPU[0]} da {self.model_path}")
                    self.pipe = ov_genai.VLMPipeline(
                        self.model_path, 
                        target_device,
                        **pipeline_kwargs
                        )
                else:
                    print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Provo a caricare il modello {self.model_name} di tipo {self.model_type} su entrambe le  {model_device_name_GPU[0]} da {self.model_path}")
                    self.pipe = ov_genai.VLMPipeline(
                        self.model_path, 
                        target_device, 
                        MODEL_DISTRIBUTION_POLICY="PIPELINE_PARALLEL",
                        **pipeline_kwargs
                        )

            self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                    )
            if target_device == "GPU":          
                print(f"\n{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Modello caricato correttamente su {model_device_name_GPU[0]}")
            else:
                print(f"\n{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Modello caricato correttamente su entrambe le {model_device_name_GPU[0]}")

        except Exception as e :
            print(f"\n{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore caricamento su {model_device_name_GPU} : {e}")
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Provo a caricare il modello {self.model_name} su {model_device_name_CPU}...")

            cpu_kwargs = {
                "KV_CACHE_PRECISION": self.kv_cache_quantization,
                "DYNAMIC_QUANTIZATION_GROUP_SIZE": 32,
            }

            if self.cache_eviction_enabled:
                cpu_kwargs["EVICTION_CONFIG"] = ov_genai.CacheEvictionConfig(
                    256,
                    512,
                    4096,
                    aggregation_mode=ov_genai.AggregationMode.NORM_SUM,
                )

            self.pipe = ov_genai.LLMPipeline(
                self.model_path, 
                "CPU",
                **cpu_kwargs
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                                self.model_path,
                                trust_remote_code=True
                                ) 
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello caricato correttamente su {model_device_name_CPU}")     

    # Costruisce una GenerationConfig OpenVINO in base al tipo di richiesta
    def _build_generation_config(self, max_new_tokens: int, is_chat: bool) -> "ov_genai.GenerationConfig":
        """
        Crea e configura un oggetto GenerationConfig riutilizzabile.
        Centralizza i parametri di decoding per evitare duplicazioni tra
        _collect_generation e stream_generator.

        Se il draft model è attivo (self.draft_pipe non None), inserisce anche
        i parametri di tree search per lo speculative decoding.

        Args:
            max_new_tokens: Limite massimo di token da generare.
            is_chat:        True per chat (sampling), False per completions (greedy).

        Returns:
            Oggetto ov_genai.GenerationConfig pronto per pipe.generate().
        """
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens

        if not is_chat:
            # Completamento: decoding deterministico (greedy)
            config.do_sample = False
            config.temperature = 0.0
            config.presence_penalty = 1.5
        else:
            # Chat: sampling con penalità anti-ripetizione
            config.do_sample = True
            config.temperature = 1.0
            config.top_p = 0.95
            config.top_k = 20
            config.min_p = 0.0
            config.presence_penalty = 1.5
            config.repetition_penalty = 1.0

        #Non funzionante con OpenVino al momento
        
        # Speculative decoding / Tree Search: abilitato solo se il draft model è caricato
        #if self.draft_pipe is not None:
            #config.do_sample = False
            #config.num_assistant_tokens = self.num_assistant_tokens
            #config.branching_factor = self.branching_factor
            #config.tree_depth = self.tree_depth
            #print(f"{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Tree search: assistant_tokens={self.num_assistant_tokens}, "
                  #f"branching_factor={self.branching_factor}, tree_depth={self.tree_depth}")

        return config

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
                config = self._build_generation_config(max_new_tokens, is_chat)
                self.pipe.generate(prompt, generation_config=config, streamer=ov_streamer)
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
        print(f"{ColoreLog.DEBUG}[STREAM]{ColoreLog.RESET} max_new_tokens = {max_new_tokens} | kwargs keys = {list(kwargs.keys())}")

        lock_acquired = self.model_lock.acquire(blocking=False)

        if not lock_acquired:
            error_payload = {
                "error": {
                    "message": "GPU busy, blocked",
                    "type": "server_error",
                    "code": "gpu_busy"
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
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
                    config = self._build_generation_config(max_new_tokens, is_chat)
                    self.pipe.generate(prompt, generation_config=config, streamer=ov_streamer)

                except Exception as e :
                    print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore generazione: {e}")

                finally : 
                    token_queue.put(None)
        
            thread = threading.Thread(target=run_generation)
            thread.start()

            try :
                found_and_think = False
                token_count_think = 0
                token_count_risposta = 0
                think_buffer = ""
                full_response_text = ""
                start_time = time.time()
                ttft = start_time
                print(f"\n{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Generazione in corso...")

                BATCH_SIZE = 4
                batch_buffer = ""

                while True :
                    try :
                        token = token_queue.get(timeout=2.0)
                    except Empty:
                        if stop_event.is_set() :
                            break
                        continue

                    if token is None :
                        if batch_buffer:
                            chunk = {
                                "choices": [{
                                    "delta": {
                                        "content": batch_buffer
                                    },
                                    "index": 0
                                }] if is_chat else {
                                    "text": batch_buffer,
                                    "index": 0
                                }
                            }

                            yield f"data: {json.dumps(chunk)}\n\n"
                            batch_buffer = ""


                        response_time = time.time() - ttft
                        token_count_risposta = len(self.tokenizer.encode(full_response_text))
                        rate = token_count_risposta / response_time if response_time > 0 else 0
                        print(f"\n{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Generazione completata in {response_time:.2f} secondi. Rate: {rate:.2f} token/s. Token generati: {token_count_risposta}\n")
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
                        _display = think_buffer.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                        print(f"\r{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Pensiero: {_display}", end="", flush=True)
                        
                        if "</think>" in think_buffer:
                            found_and_think = True
                            after_think = think_buffer.split("</think>", 1)[-1]

                            think_text = think_buffer.split("</think>", 1)[0]
                            token_count_think = len(self.tokenizer.encode(think_text))

                            ttlt_think = time.time() - start_time            

                            print(f"\n{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Pensiero completato in {ttlt_think:.2f} secondi. Token: {token_count_think}. Rate: {token_count_think / max(ttlt_think, 0.001):.1f} token/s.")
                            print(f"\n{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Token di chiusura: {repr(token)} was_thinking: {found_and_think} think_buffer: {repr(think_buffer)}")
                            
                            think_buffer = ""

                            if not after_think.strip():
                                ttft = time.time()
                                continue

                            token = after_think
                            ttft = time.time()
                        else:
                            continue

                    full_response_text += token

                    batch_buffer += token

                    if len(batch_buffer) >= BATCH_SIZE:
                        chunk = {
                            "choices": [{
                                "delta": {
                                    "content": batch_buffer
                                },
                                "index": 0
                            }] if is_chat else {
                                "text": batch_buffer,
                                "index": 0
                            }
                        }

                        yield f"data: {json.dumps(chunk)}\n\n"
                        batch_buffer = ""                

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
    def tool_stream_generator(self, message: list, prompt: str, max_new_tokens: int, use_client_tool: bool):
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
            error_payload = {
                "error": {
                    "message": "GPU busy, blocked",
                    "type": "server_error",
                    "code": "gpu_busy"
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            return

        try:
            current_message = list(message)
            current_prompt = prompt
            MAX_TOOL_CALLS = 6

            for attempt in range (MAX_TOOL_CALLS + 1):
                print(f"{ColoreLog.INFO}[TOOL_STREAM]{ColoreLog.RESET} Tentativo {attempt + 1}/{MAX_TOOL_CALLS + 1} — Generazione in corso...")
                raw_output = self._collect_generation(current_prompt, max_new_tokens, is_chat=True)
                print(f"{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Raw output ({len(raw_output)} chars): {repr(raw_output[:200])}")

                clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
                print(f"{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Clean output dopo rimozione <think>: {repr(clean_output[:200])}")

                tool_match = re.search(r"<tool_call>(.*?)</tool_call>", clean_output, flags=re.DOTALL)
                
                if tool_match and attempt < MAX_TOOL_CALLS:
                    tool_match_str = tool_match.group(1).strip()
                    print(f"{ColoreLog.INFO}[TOOL_STREAM]{ColoreLog.RESET} Tool call rilevata: {repr(tool_match_str[:200])}")

                    try:
                        func_match = re.search(r"<function=([^>]+)", tool_match_str)
                        func_name = func_match.group(1) if func_match else None

                        param_pattern = re.compile(r"<parameter=([^>]+)>\n(.*?)\n</parameter>", re.DOTALL)
                        arguments = {}
                        for match in param_pattern.finditer(tool_match_str):
                            param_name = match.group(1).strip()
                            param_value = match.group(2).strip()
                            arguments[param_name] = param_value

                        if use_client_tool:
                            call_id = f"call_{uuid.uuid4().hex[:24]}"
                            print(f"{ColoreLog.INFO}[TOOL_STREAM]{ColoreLog.RESET} Inoltro tool_call al client: {func_name} ({arguments})")

                            tool_call_chunk = {
                                "choices": [{
                                    "delta": {
                                        "tool_calls": [{
                                            "index": 0,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": func_name,
                                                "arguments": json.dumps(arguments)
                                            }
                                        }]
                                    },
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(tool_call_chunk)}\n\n"

                            finish_chunk = {
                                "choices": [{
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": "tool_calls"
                                }]
                            }
                            yield f"data: {json.dumps(finish_chunk)}\n\n"
                            return

                        print(f"{ColoreLog.INFO}[TOOL]{ColoreLog.RESET} Chiamata {func_name} ({arguments})")
                        result = execute_tool(func_name, arguments)
                        print(f"{ColoreLog.SUCCESS}[TOOL]{ColoreLog.RESET} Risultato: {result[:120]}")

                        current_message.append({"role": "assistant", "content": raw_output})
                        current_message.append({"role": "tool", "content": result, "name": func_name})

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

                print(f"{ColoreLog.SUCCESS}[DEBUG]{ColoreLog.RESET} Risposta finale ({len(final_text)} chars): {repr(final_text[:200])}")

                CHUNK_SIZE = 20
                text_chunks = [final_text[i:i + CHUNK_SIZE] for i in range(0, len(final_text), CHUNK_SIZE)] or [""]

                for idx, chunk_text in enumerate(text_chunks):
                    is_last = (idx == len(text_chunks) - 1)
                    chunk = {
                        "choices": [{
                            "delta": {"content": chunk_text}, 
                            "index": 0,
                            "finish_reason": "stop" if is_last else None
                        }]
                    }
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
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls"):
                        func = tc.get("function", {})
                        args = func.get("arguments")
                        if isinstance(args, str):
                            try:
                                func["arguments"] = json.loads(args)
                            except json.JSONDecodeError:
                                func["arguments"] = {}

            client_tools = data.get("tools")
            use_client_tool = False

            print(f"{ColoreLog.DEBUG}[ROUTE]{ColoreLog.RESET} keys ricevute dal client: {list(data.keys())}")
            print(f"{ColoreLog.DEBUG}[ROUTE]{ColoreLog.RESET} tools presente: {'tools' in data} | tool_choice: {'tool_choice' in data}")
            #print(f"{ColoreLog.DEBUG}[ROUTE]{ColoreLog.RESET} messages: {messages}")
            
            if client_tools:
                schemas = client_tools
                use_client_tool = True
            else:
                schemas = get_schemas()

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=schemas,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="medium"
            )
            
            if schemas:
                return StreamingResponse(
                    self.tool_stream_generator(
                        messages,
                        prompt,
                        max_new_tokens=4096,
                        use_client_tool=use_client_tool
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
        print(f"\n{ColoreLog.SUCCESS}[READY]{ColoreLog.RESET} Server Jarvis attivo su https://{host}:{port}\n")
        uvicorn.run(self.app, host=host, port=port)