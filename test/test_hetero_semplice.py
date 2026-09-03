"""
Script minimale per riprodurre il crash di ov_genai.VLMPipeline in modalità
HETERO:GPU.0,GPU.1 + PIPELINE_PARALLEL, da lanciare sotto GDB per catturare
il backtrace C++ completo al momento del segfault.
 
Uso:
  gdb -ex "set follow-fork-mode child" -ex "run" --args python repro_vlm_crash.py <model_path>
 
Quando crasha (SIGSEGV), dentro gdb:
  bt full
  thread apply all bt
"""
 
import sys
import openvino_genai as ov_genai
 
 
def main(model_path: str):
    device = "HETERO:GPU.0,GPU.1"
 
    print(f"Loading VLMPipeline from '{model_path}' on {device}...")
    pipe = ov_genai.VLMPipeline(
        model_path,
        device,
        MODEL_DISTRIBUTION_POLICY="PIPELINE_PARALLEL"
    )
    print("Model loaded successfully.\n")
 
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 50
    config.do_sample = False
    config.apply_chat_template = False
 
    prompt = "<|im_start|>user\nciao chi sei?<|im_end|>\n<|im_start|>assistant\n"
 
    print("Starting generation (this is where the crash is expected)...")
    result = pipe.generate(prompt, generation_config=config)
    print(f"\nGeneration completed: {result}")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python repro_vlm_crash.py <path_al_modello>")
        sys.exit(1)
 
    main(sys.argv[1])