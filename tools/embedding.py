import numpy
import os
from lettura_file import get_all_files, select_file_list, read_file
from utillities.color_logger import ColorLog

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
            selected_file = select_file_list(file_list)
            input_text = read_file(selected_file)
            return input_text
    except Exception as e:
        print(f"{ColorLog.ERROR}[ERROR]{ColorLog.RESET} Rievato errore: {e}")