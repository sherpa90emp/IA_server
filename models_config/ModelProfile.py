from dataclasses import dataclass

# ModelProfile base per modello Qwen3.8 27B
@dataclass
class ModelProfile:

    # --- KV-Cache configurazione ---
    cache_eviction_enabled: bool = True
    kv_cache_quantization: str = "f16"

    cache_size: int = 8
    max_num_seqs: int = 4
    dynamic_split_fuse: bool = True
    enable_prefix_caching: bool = True

    # Chat: sampling con penalità anti-ripetizione
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0

    start_size: int = 256               
    recent_size: int = 512              
    max_cache_size: int = 65536            
    aggregation_mode: str = "NORM_SUM"
    apply_rotation: bool = True
    snapkv_window_size: int = 8

    max_new_tokens: int = 16384

    @classmethod
    def scelta_profile(cls, model_name: str) -> "ModelProfile":
        if "Qwen3.8" in model_name:
            return cls()
        else:
            return cls(cache_eviction_enabled=False)