import numpy
import sys
import os
import tools
from tools.lettura_file import get_all_files, select_file, read_file
from utilities.color_logger import ColoreLog
from utilities.general_func import compose_path, print_all_contents, check_folder

def generate_embedding(emb_name, emb_model, emb_tokenizer, chunks):
    """
    Genera vettori di embedding per un testo fornito.

    Parametri:
        emb_name (str): Nome del modello di embedding.
        emb_model: Modello di embedding (OVModelForFeatureExtraction).
        emb_tokenizer: Tokenizer associato al modello.
        chunks (str o list): Testo da processare (può essere una stringa o lista di stringhe).

    Ritorna:
        dict: Risultato con vettori di embedding nel formato OpenAI.
        None: In caso di errore o input non valido.
    """

    if chunks is None:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Nessun testo da processare.")
        return None

    if isinstance(chunks, str): 
        chunks = [chunks]

    inputs = emb_tokenizer(
        chunks,
        padding=True,
        truncation=True,
        return_tensors="pt"
        )
    
    outputs = emb_model(**inputs)

    embeddings_list = outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": emb, "index": i} 
            for i, emb in enumerate(embeddings_list)
        ],
        "model": emb_name,
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }

def select_file_for_emb():
    """
    Seleziona un file tramite input utente per l'embedding.
    
    Funzionalità:
        - Naviga il filesystem per selezionare un file.
        - Gestisce l'uscita anticipata con "exit" o "esci".
        - Valida i percorsi e seleziona un file da elaborare.
    
    Ritorna:
        str: Contenuto del file selezionato.
        None: In caso di errore o annullamento.
    """

    try:
        current_dir = tools.lettura_file.FILE_DIR
        print(f"\n{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET}Inserire la cartella o sottocartella in cui si trova il file: \n")
        print_all_contents(current_dir)

        while True:
            user_dir = input()

            if user_dir.lower() in ["exit", "esci"]:
                sys.exit()

            folder_dir = compose_path(current_dir, user_dir)
            print(f"\n{ColoreLog.DEBUG}[DEBUG]{ColoreLog.RESET} Percorso selezionato: {folder_dir}\n")

            if not os.path.exists(folder_dir):
                print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Path inesistente. Inserire un path valido.")
                continue

            check = check_folder(folder_dir)

            if check:
                current_dir = folder_dir
                print(f"\n{ColoreLog.INPUT}[INPUT]{ColoreLog.RESET} Inserire la cartella o sottocartella in cui si trova il file: \n")
                print_all_contents(current_dir)
                continue
            else:
                file_list =  get_all_files(user_dir)
                selected_file = select_file(file_list)
                selected_file_path = compose_path(folder_dir, selected_file)
                print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Percorso del file selezionato: {selected_file_path}")
                input_text = read_file(selected_file_path)
                return input_text
                       
    except Exception as e:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Rilevato errore: {e}")
        return None

def chunk_testo(input_text: str, emb_tokenizer, max_token: int, overlap: int):
    """
    Suddivide un testo in chunk di dimensioni massime definite.
    
    Parametri:
        input_text (str): Testo da suddividere.
        emb_tokenizer: Tokenizer per calcolare i token.
        max_token (int): Dimensione massima per ciascun chunk.
        overlap (int): Sovrapposizione tra i chunk.
    
    Ritorna:
        list: Lista di chunk.
        None: Se il testo è vuoto.
    """

    if input_text is None:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Nessun testo rilevato.")
        return None

    tokenizer_input_text = len(emb_tokenizer.encode(input_text))

    if tokenizer_input_text < max_token:
        chunk = [input_text]
        return chunk
    else:
        chunks_list = []
        token_ids = emb_tokenizer.encode(input_text)
        for i in range(0, len(token_ids), (max_token - overlap)):
            chunk_ids = token_ids[i : i + max_token]
            chunk_decoded_testo = emb_tokenizer.decode(chunk_ids)
            chunks_list.append(chunk_decoded_testo)
        return chunks_list    
