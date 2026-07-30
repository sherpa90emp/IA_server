import numpy
import os
from tools.lettura_file import get_all_files, select_file, read_file
from utilities.color_logger import ColoreLog

def generate_embedding(emb_name, emb_model, emb_tokenizer, input_text):
    if isinstance(input_text, str): 
        input_text = [input_text]

    inputs = emb_tokenizer(
        input_text,
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
        print("In quale cartella si trova il file?")
        user_dir = input()
        if user_dir:
            file_list =  get_all_files(user_dir)
            selected_file = select_file(file_list)
            input_text = read_file(selected_file)
            return input_text
    except Exception as e:
        print(f"{ColoreLog.ERROR}[ERROR]{ColoreLog.RESET} Rievato errore: {e}")

def chunk_testo(input_text: str, emb_tokenizer, max_token: int, overlap: int):
    tokenizer_input_text = len(emb_tokenizer.encode(input_text))
    if tokenizer_input_text < max_token:
        input_text = [input_text]
        return input_text
    else:
        input_text_list = []
        token_ids = emb_tokenizer.encode(input_text)
        for i in range(0, len(token_ids), (max_token - overlap)):
            chunk_ids = token_ids[i : i + max_token]
            chunk_decoded_testo = emb_tokenizer.decode(chunk_ids)
            input_text_list.append(chunk_decoded_testo)
        return input_text_list    
