import os
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
from utilities.color_logger import ColoreLog
from utilities.general_func import rileva_device

def conferma_uso_emb():
    """
    Conferma l'uso dei modelli di embedding e richiama la funzione per selezionarli.
    
    Funzionalità:
        - Stampa un messaggio di conferma.
        - Delega alla funzione `get_local_models_emb()` per la selezione del modello.
    """

    print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Procedo alla selezione dei modelli di embedding")
    return get_local_models_emb()

def get_local_models_emb():
    """
    Cerca i modelli di embedding locali e li elenca per selezione.
    
    Funzionalità:
        - Verifica l'esistenza della cartella dei modelli.
        - Raccoglie i modelli locali con "emb" nel nome.
        - Se presenti, li elenca e chiede all'utente di selezionarne uno.
        - Se non presenti, carica il modello predefinito da Hugging Face.
    
    Ritorna:
        tuple: (nome_modello, percorso_modello) o (None, None) in caso di errore.
    """
    
    model_dir = "/home/andrea/models"
    default_name = "Qwen3-Embedding-0.6B-int4-cw-ov"
    repo_id = f"OpenVINO/{default_name}"

    if not os.path.exists(model_dir):
        print( f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Cartella {model_dir} non trovata.")
        return None, None
    
    local_models_emb = [
        m for m in os.listdir(model_dir)
        if "emb" in m.lower() and os.path.isdir(os.path.join(model_dir, m))
    ]

    if local_models_emb:
        print("\nModelli già presenti localmente:\n")

        for i, m in enumerate(local_models_emb):
            print(f"{i+1} - {m}")

        while True:
            try:
                scelta = int(input(f"\n{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET} Quale modello di embedding vuoi usare? ")) -1

                if 0 <= scelta < len(local_models_emb):
                    selected_name = local_models_emb[scelta]
                    selected_path = os.path.join(model_dir, selected_name)
                    return selected_name, selected_path
                else:
                    print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Scelta non valida.")  

            except ValueError:
                print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Inserisci un numero presente nell'elenco.")

    else:
        print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Non sono stati trovati modelli custom di embedding")
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Procedo al caricamento del modello predefinito {default_name}")

        default_path = os.path.join(model_dir, default_name)

        if not os.path.exists(default_path):
            print(f"\n{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Modello predefinito non trovato localmente.")
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Download da Hugging Face in corso...")

            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=default_path
                )
                print(f"\n{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Download completato.")

            except Exception as e:
                print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Download fallito: {e}")
                return None, None
            
        return default_name, default_path

def load_model_emb(emb_name, emb_path):
    """
    Carica il modello di embedding e il tokenizer associato.
    
    Parametri:
        emb_name (str): Nome del modello.
        emb_path (str): Percorso locale del modello.
    
    Funzionalità:
        - Rileva il dispositivo (GPU o CPU).
        - Carica il modello con `OVModelForFeatureExtraction`.
        - Carica il tokenizer con `AutoTokenizer`.
    
    Ritorna:
        tuple: (emb_model, emb_tokenizer) o (None, None) in caso di errore.
    """
    
    model_device_name_GPU, model_device_name_CPU, target_device = rileva_device()

    if emb_path and os.path.exists(emb_path):
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Caricamento modello Embedding {emb_name} sulla {model_device_name_GPU} da {emb_path}")
        emb_model = OVModelForFeatureExtraction.from_pretrained(
            emb_path,
            device=target_device
        )

        emb_tokenizer = AutoTokenizer.from_pretrained(emb_path)

        print(f"{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Modello caricato correttamente su {model_device_name_GPU}") 
        return emb_model, emb_tokenizer
    
    else:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Caricamento del modello interrotto.")
        return None, None    