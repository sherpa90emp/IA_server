import numpy

def generate_embedding(emb_model, emb_tokenizer, input_text):
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