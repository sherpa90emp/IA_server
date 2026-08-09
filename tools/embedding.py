import numpy
import  tools
from tools.lettura_file import get_all_files, select_file, read_file
from utilities.color_logger import ColoreLog
from utilities.general_func import compose_path

def generate_embedding(emb_name, emb_model, emb_tokenizer, chunks):
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
    try:
        print("\nIn quale cartella si trova il file?\n")
        user_dir = input()
        if user_dir:
            file_list =  get_all_files(user_dir)
            selected_file = select_file(file_list)
            folder_dir = compose_path(tools.lettura_file.FILE_DIR, user_dir)
            selected_file_path = compose_path(folder_dir, selected_file)
            print(f"{ColoreLog.INFO}[INFO]{ColoreLog.RESET} Percorso del file selezionato: {selected_file_path}")
            input_text = read_file(selected_file_path)
            return input_text
    except Exception as e:
        print(f"{ColoreLog.ERRORE}[ERROR]{ColoreLog.RESET} Rilevato errore: {e}")
        return None

def chunk_testo(input_text: str, emb_tokenizer, max_token: int, overlap: int):
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
