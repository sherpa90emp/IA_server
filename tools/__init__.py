from tools.meteo import get_meteo, formatta_meteo
from tools.lettura_file import search_file, read_file, mod_file
from utilities.color_logger import ColoreLog

# Registry centrale: nome_tool → { func, schema }
TOOL_REGISTRY: dict = {}

# Registra un tool nel registry globale con la sua funzione e schema
def register_tool(name: str, func, schema: dict) -> None:
    """
    Registra un tool nel registry globale.

    Args:
        name:   Nome del tool (deve corrispondere a quello nel JSON schema).
        func:   Funzione Python da eseguire.
        schema: Schema OpenAI-compatible del tool (type, function, parameters).
    """
    TOOL_REGISTRY[name] = {"func": func, "schema": schema}
    print(f"{ColoreLog.SUCCESS}[TOOL]{ColoreLog.RESET} Registrato: {name}")

# Esegue un tool per nome con gli argomenti forniti dal modello
def execute_tool(name: str, arguments: dict) -> str:
    """
    Esegue un tool per nome con gli argomenti forniti dal modello.

    Args:
        name:      Nome del tool da eseguire.
        arguments: Dizionario degli argomenti estratti dal tool_call.

    Returns:
        Risultato come stringa da reinserire nel contesto del modello.
    """
    if name not in TOOL_REGISTRY:
        return f"Errore: tool '{name}' non trovato nel registry."
    try:
        result = TOOL_REGISTRY[name]["func"](**arguments)
        # Se il risultato è un dict (es. get_meteo), lo formatto in testo
        if isinstance(result, dict):
            if "temperatura" in result:
                return formatta_meteo(result)
            return str(result)
        elif isinstance(result, list):
            result_list = "File trovato:\n" + "\n".join(result)
            return result_list
        return str(result)
    except Exception as e:
        return f"Errore esecuzione tool '{name}': {e}"

# Restituisce la lista degli schemi di tutti i tool registrati
def get_schemas() -> list:
    """
    Restituisce la lista degli schemi di tutti i tool registrati.
    Da passare a apply_chat_template come argomento tools=.
    """
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


# ---------------------------------------------------------------------------
# Registrazione tool disponibili
# ---------------------------------------------------------------------------

register_tool(
    name="get_meteo",
    func=get_meteo,
    schema={
        "type": "function",
        "function": {
            "name": "get_meteo",
            "description": "Recupera le condizioni meteo attuali per una città italiana o estera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "citta": {
                        "type": "string",
                        "description": "Nome della città, es. 'Roma', 'Milano', 'Firenze', 'Londra'"
                    }
                },
                "required": ["citta"]
            }
        }
    }
)

register_tool(
    name="search_file",
    func=search_file,
    schema={
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Recupera un file specifico o un gruppo di file simili all'interno di una dir specifica",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Nome del file, es. 'riassunto_video.txt', 'dati.txt'"
                    },
                    "subdir": {
                        "type": "string",
                        "description": "Nome della subdir, può essere None"
                    },
                },
                "required": ["filename"]
            }
        }
    }
)

register_tool(
    name="read_file",
    func=read_file,
    schema={
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Legge il contenuto di un file specifico fornito come argomento. Da utilizzare sempre prima di chiamare mod_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Percorso del file, es. '/path/to/file.txt'"
                    },
                },
                "required": ["file_path"]
            }
        }
    }
)

register_tool(
    name="mod_file",
    func=mod_file,
    schema={
        "type": "function",
        "function": {
            "name": "mod_file",
            "description": "Modifica il contenuto di un file specifico fornito come argomento sulla base del contenuto fornito come argomento. IMPORTANTE: prima di chiamare questo metodo e modificare il file, e' necessario chiedere la CONFERMA all'utente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Percorso del file, es. '/path/to/file.txt'"
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenuto del file."
                    },
                },
                "required": ["file_path", "content"]
            }
        }
    }
)