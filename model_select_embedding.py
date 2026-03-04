import os

def get_local_models_emb():
    model_dir = "../models"
    if not os.path.exists(model_dir):
        return []
    
    local_models_emb = []

    for m in os.listdir(model_dir):
        if "enb" in m.lower and os.isdir(os.path.join(model_dir, m)):
            local_models_emb.append(m)
    return sorted(local_models_emb)