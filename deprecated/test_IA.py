import torch
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import intel_extension_for_pytorch as ipex

# Nome di un modello leggero per il test
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Caricamento modello su Arc Pro B50 con ipex-llm...")

# Carichiamo il modello quantizzato a 4-bit (ottimizzato per Intel)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    optimize_model=True,
    trust_remote_code=True,
    use_cache=True
)

# Spostiamo il modello sulla GPU (XPU)
model = model.to('xpu')

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Test di generazione
prompt = "Ciao, chi sei?"
inputs = tokenizer(prompt, return_tensors="pt").to('xpu')

print("Generazione in corso...")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=20)
    
print("-" * 20)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("-" * 20)