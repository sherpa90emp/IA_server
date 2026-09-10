import os
import sys
import json
from utilities.color_logger import ColoreLog
from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVModelForCausalLM, OVModelForVisualCausalLM
from optimum.exporters.openvino.convert import export_tokenizer
from transformers import AutoTokenizer, AutoConfig
from utilities.general_func import recupero_dimensione_modello_dal_nome

# Ottiene l'elenco dei modelli locali disponibili nella directory
def get_local_models():
    """
    Scansiona la directory dei modelli locali e restituisce una lista ordinata dei modelli disponibili.
    
    Ritorna:
        list: Lista di nomi di modelli locali (vuota se nessun modello è presente).
    """
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
    """
    Stampa un messaggio iniziale che mostra i modelli disponibili e richiede all'utente un input.
    
    Args:
        local_models (list): Lista di modelli locali da visualizzare.
    """

    if local_models:
        print("\nModelli già presenti localmente:")

        for i, model in enumerate(local_models, 1):
            print(f"{i} - {model}")

    print(f"\n{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET}Inserisci il numero o il nome del modello che desideri usare: ")
    print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Premendo INVIO verrà usato il modello predefinito.")
    print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET}Scrivi EXIT per uscire.\n")

# Funzione per stampare un messaggio di avviso in caso di errore
def messaggio_next_error():
    """
    Stampa un messaggio di avviso quando il modello selezionato non è presente nei repository di Huggingface.
    """

    print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Il modello selezionato non era presente nei repository di Huggingface.\n")
    print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Inserire un modello corretto\n")

# Funzione per ottenere l'input dell'utente
def get_user_input(local_models):
    """
    Gestisce l'input dell'utente per selezionare un modello, con supporto per l'uscita, il modello predefinito o la selezione per numero/nome.
    
    Args:
        local_models (list): Lista di modelli locali disponibili.
    
    Returns:
        str: Nome del modello selezionato o predefinito.
    """

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
                print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Numero non valido, inserisci quello corretto.\n")

        return user_input

# Verifica se il modello esiste localmente, altrimenti lo scarica o lo converte
def check_and_prepare_model(model_name, model_path):
    """
    Verifica se il modello esiste localmente. Se non esiste, chiede all'utente di scaricarlo o convertirlo.
    
    Args:
        model_name (str): Nome del modello.
        model_path (str): Percorso locale dove salvare il modello.
    
    Returns:
        tuple: (model_name, model_path, model_type) ad operazione riuscita, None altrimenti.
    """

    if not os.path.exists(model_path) :
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello non trovato in {model_path} \n")
        
        confirm = input(f"{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET}\nVuoi scaricarlo/esportarlo ora (s/n): ")

        if confirm.lower() != 's' :
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Operazione annullata. Inserisci un altro modello.\n")
            return None
                    
        if "OpenVINO" in model_name or "-ov" in model_name :
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Scaricamento del modello {model_name} ottimizzato da Huggingface...\n")
            snapshot_download(model_name, local_dir=model_path)
            print(f"\n{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Download completato.\n")
        else :
            print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello OpenVINO non trovato. Avvio procedura di esportazione per {model_name}\n")

            model_hf_type = check_model_type_from_hf(model_name)

            if model_hf_type is None :
                print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Impossibile determinare il tipo di modello da HF.\n")
                return None

            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello {model_hf_type.upper()}. Avvio procedura di esportazione per {model_name}...\n")

            conversion_model(model_name, model_path, model_hf_type)

        model_type = check_type_model(model_path)

        if model_type is None :
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Operazione annullata. Tipo di modello non riconosciuto.\n")
            return None
        else:
            return model_name, model_path, model_type
        
    else :
        print(f"\n{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello {model_name} già presente localmente. Procedo al caricamento...\n")
        model_type = check_type_model(model_path)

        if model_type is None :
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Operazione annullata. Tipo di modello non riconosciuto.\n")
            return None
        else:
            return model_name, model_path, model_type


# Funzione principale per selezionare il modello
def get_model_selection() :
    """
    Funzione principale per la selezione del modello. Gestisce il ciclo di selezione, gestione errori e richiama altre funzioni.
    
    Returns:
        tuple: (model_name, model_path, model_type) ad operazione riuscita.
    """

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
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Errore durante la selezione del modello: {e}\n")
            errore_rilevato = True

def check_type_model(model_path) :
    """
    Determina il tipo di modello (llm o vlm) analizzando il file config.json.
    
    Args:
        model_path (str): Percorso del modello.
    
    Returns:
        str: Tipo di modello ('llm' o 'vlm') o None se non riconosciuto.
    """
    
    type_model_path = os.path.join(model_path, "config.json")

    
    if not os.path.exists(type_model_path) :
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Path inesistente. Inserire un path valido.\n")
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
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Key inesistente: {e}\n")
            return None

def check_model_type_from_hf(model_name) :
    """
    Determina il tipo di modello (LLM/VLM) analizzando il config.json del modello su Hugging Face.
    
    Args:
        model_name (str): Nome del modello su Hugging Face.
    
    Returns:
        str: 'llm' o 'vlm' in base all'architettura, None se non riconosciuto.
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        arch_str = architectures[0]

        if "ForConditionalGeneration" in arch_str or "VL" in arch_str:
            return "vlm"
        else:
            return "llm"
            
    except Exception as e:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Impossibile ottenere il tipo di modello da Hugging Face: {e}\n")
        return None

def conversion_model(model_name, model_path, model_hf_type) :
    """
    Conversione di un modello di Hugging Face in formato OpenVINO.
    
    Args:
        model_name (str): Nome del modello su Hugging Face.
        model_path (str): Percorso dove salvare il modello OpenVINO.
    """
    quantization_config = richiesta_quantization_config(model_hf_type)

    if model_hf_type == "vlm":
        ov_model_for = OVModelForVisualCausalLM
    else:
        ov_model_for = OVModelForCausalLM

    ov_model =  ov_model_for.from_pretrained(
        model_name,
        export=True,
        compile=False,
        load_in_8bit=False,
        fix_mistral_regex=True,
        quantization_config=quantization_config
    )

    ov_model.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)
    export_tokenizer(tokenizer, model_path)
    
    print(f"{ColoreLog.SUCCESS}[SUCCESS]{ColoreLog.RESET} Conversione completata. Modello salvato in: {model_path}\n")
    del ov_model


def richiesta_quantization_config(model_hf_type) :
    print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Inizio procedura settaggio parametri di conversione del tipo di modello {str(model_hf_type).upper()} \n")

    bits_model = [4, 8]
    sym_model = [True, False]
    group_size_model = [64, 128]
    ratio_model = [0.5, 0.8]

    bits = richiesta_valori(bits_model, bits_model[0], int)
    sym = richiesta_valori(sym_model, True, bool)
    group_size = richiesta_valori(group_size_model, group_size_model[0], int)
    ratio = richiesta_valori(ratio_model, ratio_model[0], float)

    quantization_config = {
        "bits": int(bits),
        "sym": bool(sym),
        "group_size": int(group_size),
        "ratio": float(ratio)
    }

    return quantization_config

def stampa_richiesta_valori_quant_config(parametro, valore):
    print(f"{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET} Inserisci il valore, es {parametro}:\n")
    print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Premere INVIO per usare il valore di default: {valore}\n")

def richiesta_valori(valori_ammessi, valore_default, tipo_conversione):
    while True:
        stampa_richiesta_valori_quant_config(valori_ammessi, valore_default)
        valore_input = input()

        if not valore_input:
            return valore_default

        try:
            if tipo_conversione == bool:
                if valore_input.lower() == "true":
                    valore = True
                elif valore_input.lower() == "false":
                    valore = False
                else:
                    raise ValueError("Valore booleano non riconosciuto. Inserire True o False o premere invio per utilizzare il valore di default.\n")
                return valore
            else:    
                valore = tipo_conversione(valore_input)
                if valore in valori_ammessi:
                    return valore
                else:
                    print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Valore non valido. Inserire un valore valido.\n")
        except ValueError as e:
            print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Input non valido. Inserire un valore valido.\n Error: {e}\n")

def load_draft_model():

    model_disponibili = get_local_models()
    candidati = []
    max_draft_params = 3

    for model in model_disponibili:
        model_path = f"/home/andrea/models/{model}"

        if not os.path.exists(model_path):
            continue

        params = recupero_dimensione_modello_dal_nome(model)

        if params is None:
            print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Parametri non trovati per il modello {model}\n")
            continue

        if params <= max_draft_params:
            candidati.append((model, model_path, params))

    if not candidati:
        print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Nessun modello con parametri inferiore o uguale a {max_draft_params}B trovato.\n")
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Speculative decoding non disponibile.\n")
        return None

    if len(candidati) == 1:
        name, path, params_draft = candidati[0]
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello draft selezionato: {name} con {params_draft}B di parametri\n")
        confirm = input(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Usare questo modello? (s/n): \n")

        if confirm.lower() == 's':
            return name, path
        else:
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Speculative decoding disattivato.\n")
            return None, None
    else:
        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Seleziona il modello draft da usare:\n")

        for i, (name, path, params_draft) in enumerate(candidati):
            print(f"{i+1} - {name} con {params_draft}B di parametri")

        print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Premere INVIO per disattivare speculative decoding.")

        while True:
            user_input = input()

            if not user_input:
                print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Nessun modello selezionato.\n")
                print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Speculative decoding disattivato.\n")
                return None, None

            if user_input.isdigit():
                idx = int(user_input) - 1

                if 0 <= idx < len(candidati):
                    name, path, params_draft = candidati[idx]
                    print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Modello draft selezionato: {name} con {params_draft}B di parametri\n")
                    return name, path
                else:
                    print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Numero non valido.\n")
            else:
                print(f"{ColoreLog.WARNING}[WARNING]{ColoreLog.RESET} Input non valido. Inserire un numero valido.\n")            