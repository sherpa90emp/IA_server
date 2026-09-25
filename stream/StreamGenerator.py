import json
import threading
import time

from utilities.color_logger import ColoreLog
from queue import Queue, Empty

class StreamGenerator:
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

        lock_acquired = self.model_lock.acquire(timeout=120)

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