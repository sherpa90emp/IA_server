import os
import sys
import json
from utilities.color_logger import ColoreLog
from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVModelForCausalLM
from optimum.exporters.openvino.convert import export_tokenizer
from transformers import AutoTokenizer

# Ottiene l'elenco dei modelli locali disponibili nella directory
def get_local_models():
    models_dir = "/home/andrea/models"
    if not os.path.exists(models_dir):
        return [] 
    
    local_models = [
        m for m in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, m))
    ]
    return sorted(local_models)

# Funzione per stampare il messaggio iniziale con i modelli disponibili
def messaggio_iniziale(local_models): 
    if local_models:
        print("\nModelli già presenti localmente:")
        for i, model in enumerate(local_models, 1):
            print(f"{i} - {model}")

    print("\nInserisci il numero o il nome del modello che desideri usare: ")
    print("\nPremendo INVIO verrà usato il modello predefinito. (Qwen3-14B-int4-ov)")
    print("\nScrivi EXIT per uscire.\n")

# Funzione per stampare un messaggio di avviso in caso di errore
def messaggio_next_error():
    print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Il modello selezionato non era presente nei repository di Huggingface.")
    print("Inserire un modello corretto")

# Funzione per ottenere l'input dell'utente
def get_user_input(local_models):
    while True:
        user_input = input().strip()

        if user_input.lower() == "exit" :
            sys.exit(0)

        if not user_input :
            return "Qwen/Qwen3-14B-int4-ov"
        if user_input.isdigit():
            i = int(user_input) - 1
            if 0 <= i < len(local_models):
                return local_models[i]
            else:
                print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Numero non valido, inserisci quello corretto")
        return user_input

# Verifica se il modello esiste localmente, altrimenti lo scarica o lo converte
def check_and_prepare_model(model_name, model_path):
    if not os.path.exists(model_path) :
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello non trovato in {model_path}")
        
        confirm = input("Vuoi scaricarlo/esportarlo ora (s/n): ")
        if confirm.lower() != 's' :
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Operazione annullata. Inserisci un altro modello.")
            return None
                    
        if "OpenVINO" in model_name or "-ov" in model_name :
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Scaricamento del modello {model_name} ottimizzato da Huggingface...")
            snapshot_download(model_name, local_dir=model_path)
            print(f"\n{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Download completato.")
        else :
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello OpenVINO non trovato. Avvio procedura di esportazione per {model_name}")
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Esportazione e quantizzazione int4 in corso (potrebbe richiedere qualche minuto)...")

            ov_model = OVModelForCausalLM.from_pretrained(
                model_name,
                export=True,
                compile=False,
                load_in_8bit=False,
                fix_mistral_regex=True,
                quantization_config={
                    "bits": 4,
                    "sym": True,
                    "group_size": 128,
                    "ratio": 0.8
                }    
            )
            ov_model.save_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)
            export_tokenizer(tokenizer, model_path)
            
            print(f"{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Conversione completata. Modello salvato in: {model_path}")
            del ov_model
        model_type = check_type_model(model_path)
        if model_type is None :
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Operazione annullata. Tipo di modello non riconosciuto.")
            return None
        else:
            return model_name, model_path, model_type
    else :
        print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello {model_name} già presente localmente. Procedo al caricamento...")
        model_type = check_type_model(model_path)
        if model_type is None :
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Operazione annullata. Tipo di modello non riconosciuto.")
            return None
        else:
            return model_name, model_path, model_type


# Funzione principale per selezionare il modello
def get_model_selection() :
    errore_rilevato = False
    while True :
        models_disponibili = get_local_models()

        if not errore_rilevato:
            messaggio_iniziale(models_disponibili)
        else:
            messaggio_next_error()

        model_name = get_user_input(models_disponibili)

        if "-ov" in model_name :
            model_path = f"/home/andrea/models/{model_name.split('/')[-1]}"
        else :
            model_path = f"/home/andrea/models/{model_name.split('/')[-1]}-ov"

        try :
            result = check_and_prepare_model(model_name, model_path)
            if result:
                return result
            else:
                errore_rilevato = False
                continue
        except Exception as e :
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore durante la selezione del modello: {e}")
            errore_rilevato = True

def check_type_model(model_path) :
    type_model_path = os.path.join(model_path, "config.json")

    
    if not os.path.exists(type_model_path) :
        print("Path inesistente. Inserire un path valido.")
        return None
    
    with open(type_model_path, "r") as f:
        try : 
            data = json.load(f)
            architectures = data["architectures"]
            arch_str = architectures[0]
            if "ForConditionalGeneration" in arch_str or "VL" in arch_str:
                model_type = "vlm"
                return model_type
            else:
                model_type = "llm"
                return model_type
        except Exception as e :
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Key inesistente: {e}")
            return None            