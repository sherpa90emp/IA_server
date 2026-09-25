import json
import datetime
import re
import uuid
import threading

from utilities.color_logger import ColoreLog
from tools import execute_tool
from queue import Queue, Empty

class ToolStreamGenerator(ColoreLog):
    # Gestisce il ciclo di chiamate tool e streaming finale
    def tool_stream_generator(self, message: list, prompt: str, max_new_tokens: int, use_client_tool: bool, disconnect_event: None):
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
        lock_acquired = self.model_lock.acquire(timeout=120)
        response_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"

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
            role_chunk = {
                        "id": response_id,
                        "model": self.model_name,
                        "choices": [
                            {
                                "delta": {
                                    "role": "assistant",
                                },
                                "index": 0
                            }
                        ]
                    }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            current_message = list(message)
            current_prompt = prompt
            MAX_TOOL_CALLS = 6

            for attempt in range (MAX_TOOL_CALLS + 1):
                if disconnect_event is not None and disconnect_event.is_set():
                    print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Client disconnesso")
                    return

                print(f"{ColoreLog.INFO}[TOOL_STREAM {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]{ColoreLog.RESET} Tentativo {attempt + 1}/{MAX_TOOL_CALLS} — Generazione in corso...")
                raw_output = self._collect_generation(current_prompt, max_new_tokens, is_chat=True)
                print(f"{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Raw output ({len(raw_output)} chars): {repr(raw_output[:200])}")

                clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
                print(f"{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Clean output dopo rimozione <think>: {repr(clean_output[:200])}")

                tool_match = re.search(r"<tool_call>(.*?)</tool_call>", clean_output, flags=re.DOTALL)
                
                if tool_match and attempt < MAX_TOOL_CALLS:
                    tool_match_str = tool_match.group(1).strip()
                    print(f"{ColoreLog.INFO}[TOOL_STREAM {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]{ColoreLog.RESET} Tool call rilevata: {repr(tool_match_str[:200])}")

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
                            call_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
                            print(f"{ColoreLog.INFO}[TOOL_STREAM]{ColoreLog.RESET} Inoltro tool_call al client: {func_name} ({arguments})")

                            tool_call_chunk = {
                                "choices": [{
                                    "delta": {
                                        "tool_calls": [{
                                            "index": 0,
                                            "id": call_id,
                                            "model": self.model_name,
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
                    final_chunk = {
                        "id": response_id,
                        "model": self.model_name,
                        "choices": [
                            {
                                "delta": {
                                    "content": chunk_text
                                    }, 
                                "index": 0,
                                "finish_reason": "stop" if is_last else None
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(self.tokenizer.encode(current_prompt)),
                            "completion_tokens": len(self.tokenizer.encode(final_text)),
                            "prompt_tokens_details": {
                                "cached_tokens": 0
                            },
                            "cost": 0
                        }
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                break

        finally:
            self.model_lock.release()

    # Esegue generazione non-streaming del modello
    def _collect_generation(self, prompt: str, max_new_tokens: int, is_chat: bool, disconnect_event=None) -> str:
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
            if stop_event.is_set() or (disconnect_event is not None and disconnect_event.is_set()):
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
        think_buffer = ""
        found_and_think = False

        while True:
            try:
                token = token_queue.get(timeout=5.0)
            except Empty:
                continue

            if token is None:
                break

            if not found_and_think:
                think_buffer += token
                _display = think_buffer.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                print(f"\r{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Pensiero: {_display}", end="", flush=True)

                if "</think>" in think_buffer:
                    found_and_think = True
                    after_think = think_buffer.split("</think>", 1)[-1]
                    token = after_think

            output += token

        thread.join()

        return output