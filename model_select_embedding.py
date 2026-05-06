import os
from huggingface_hub import snapshot_download
from color_logger import ColoreLog

def conferma_uso_emb():
    print(f"Vuoi utilizzare un modello di embedding? s/N")
    user_input = input().strip().lower()

    if user_input == "s":
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Procedo alla selezione dei modelli di embedding")
        return get_local_models_emb()
    else:
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modalità 'Solo Regole' attiva.")
        return None, None

def get_local_models_emb():
    model_dir = "../models"
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
        print("\nModelli già presenti localmente:")
        for i, m in enumerate(local_models_emb):
            print(f"{i+1} - {m}")
        while True:
            try:
                scelta = int(input("\nQuale modello di embedding vuoi usare? ")) -1
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