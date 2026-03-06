import os
from color_logger import ColoreLog

def get_local_models_emb():
    model_dir = "../models"
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
        print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Non sono stati trovati modelli custom")
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Procedo al caricamento del modello predefinito.")
        default_name = "Qwen3-Embedding-0.6B-int4-cw-ov"
        default_path = os.path.join(model_dir, default_name)
        return default_name, default_path 