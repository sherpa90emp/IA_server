## Rules
- Rispondi SEMPRE e SOLO in italiano, indipendentemente dalla lingua della domanda. È obbligatorio.
- Non rimuovere MAI gli import dall'inizio dei file.
- Quando richiesto, fornisci sempre commenti professionali ai metodi.
- In modalità Agent o Chat, quando proponi modifiche a un file esistente, NON riscrivere le parti di codice non modificate. Al loro posto inserisci un placeholder come `// ...codice esistente...` e mostra solo le righe effettivamente cambiate.
- Non introdurre dipendenze esterne senza accordo esplicito.
- I modelli si trovano in `../models/` (fuori dal repository) — non modificare mai quel percorso.
- Il server espone API OpenAI-compatibili: non alterare i path `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`.
- Non rimuovere la logica di filtraggio del tag `<think>` in `stream_generator` — è necessaria per i modelli reasoning (es. Qwen).

## Project Overview
**Jarvis** è un server AI locale che esegue LLM tramite **OpenVINO** su hardware Intel (GPU Arc B50, con fallback CPU).
Espone un'API REST compatibile con le specifiche OpenAI, usata come backend per **Continue** (autocompletamento e chat nell'IDE) e per una web chat Streamlit.

## Tech Stack
| Layer            | Tecnologia                        | Note                                  |
|------------------|-----------------------------------|---------------------------------------|
| Linguaggio       | Python                            | Nessun requirements.txt nel repo      |
| Server REST      | FastAPI + Uvicorn                 | API OpenAI-compatibile                |
| Inferenza LLM    | `openvino_genai` (LLMPipeline)   | GPU Intel Arc B50 → fallback CPU      |
| Export modelli   | `optimum.intel` (OVModelForCausalLM) | Quantizzazione int4 automatica    |
| Embedding        | `optimum.intel` (OVModelForFeatureExtraction) | Opzionale              |
| Tokenizzazione   | `transformers` (AutoTokenizer)    |                                       |
| Download modelli | `huggingface_hub`                 | `snapshot_download`                   |
| Web UI           | Streamlit                         | `server_jarvis_web_chat.py` (WIP)     |
| Logging          | `colorama`                        | Wrapper in `color_logger.py`          |

## Struttura del Repository
```
IA_server/
├── main_jarvis.py                 # Entrypoint: wizard avvio + orchestrazione
├── model_select_jarvis.py         # Selezione, download ed export modello LLM
├── model_select_embedding.py      # Selezione modello embedding
├── server_jarvis_IDE.py           # Server FastAPI (classe JarvisServerIDE)
├── server_jarvis_web_chat.py      # Web chat Streamlit (WIP)
├── color_logger.py                # Logger colorato con colorama
├── rules_IA_Server.md             # Regole base del progetto
├── tools/
│   |── meteo.py                   # Tool meteo
    |── embedding.py               # Script per tools di embedding (incompleto)
    |── lettura_file.py            # Tools per modifica e lettura file
    |── _init_.py                  # Contine la logica per l'utilizzo dei tools
├── test/                          # Script di test e benchmark
└── deprecated/                    # Script obsoleti (non modificare)

../models/                         # Modelli OpenVINO (fuori dal repo, non versionati)
```
La struttura interna può evolvere. Fai sempre riferimento ai file presenti nel contesto della conversazione per conoscere la struttura aggiornata.

## Architettura
```
[Continue IDE / Web Chat]
        ↓ HTTP streaming (SSE)
    porta 8000
        ↓
[FastAPI — JarvisServerIDE]
    POST /v1/chat/completions   → chat (max 4096 token, temp 0.6, streaming)
    POST /v1/completions        → autocompletamento (max 64 token, temp 0.0, deterministico)
    GET  /v1/models             → lista modelli
        ↓
[LLMPipeline OpenVINO — GPU Arc B50 / CPU fallback]
        ↓
[../models/{nome-modello}-ov]   → modelli quantizzati int4
```

## Convenzioni del Codice

### Gestione modelli
- I modelli locali sono in `../models/` con suffisso `-ov` (es. `Qwen2.5-Coder-1.5B-ov`)
- Se il modello non è OpenVINO nativo viene esportato e quantizzato automaticamente: `int4`, `sym=True`, `group_size=128`, `ratio=0.8`

### Classe `JarvisServerIDE`
- Un'unica istanza per processo; accesso al modello protetto da `threading.Lock()` (una richiesta alla volta)
- La generazione avviene in un thread separato con `Queue` per il token streaming
- Il `stream_generator` filtra i blocchi `<think>...</think>` prima di inviare i token al client
- Per le completions, i token di stop hardcodati sono: `"README.md"`, `"Copyright"`, `"---"`, `"repo_name"`, `"/*"`, `"*/"`

### Logging
- Usare sempre `ColoreLog` da `color_logger.py` per i messaggi di console
- Livelli: `ColoreLog.INFO`, `ColoreLog.SUCCESS`, `ColoreLog.WARNING`, `ColoreLog.ERRORE`, `ColoreLog.DEBUG`
- Formato: `print(f"{ColoreLog.LIVELLO}[LIVELLO]{ColoreLog.RESET} Messaggio")`

### Aggiunta di nuovi tool
- I tool vanno in `tools/` come moduli Python indipendenti
- Ogni tool deve essere autonomo e importabile senza side effect all'import