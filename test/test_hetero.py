"""
Script diagnostico minimale per isolare il crash HETERO:GPU.0,GPU.1
pipeline-parallel a livello di ov.Core, senza passare dal layer ov_genai.
 
Obiettivo: capire se il page fault sul motore bcs (blitter) si riproduce
anche con un modello "grezzo" compilato via core.compile_model(), o se
è specifico a come ov_genai (VLMPipeline/LLMPipeline) alloca e passa
i buffer internamente.
 
Usa un modello OpenVINO IR qualsiasi già presente sul disco (puoi puntare
a un .xml di un modello int4-ov già convertito, es. Qwen3-14B-int4-ov/openvino_model.xml)
purché sia abbastanza grande da forzare lo split reale tra le due GPU.
Se il modello è troppo piccolo, HETERO potrebbe metterlo tutto su GPU.0
senza mai coinvolgere il trasferimento dati verso GPU.1, vanificando il test.
"""
 
import sys
import openvino as ov
import openvino.properties.hint as hints
import numpy as np
 
 
def main(model_xml_path: str, n_runs: int = 20):
    core = ov.Core()
 
    print(f"Available devices: {core.available_devices}")
 
    print(f"\nCompiling model '{model_xml_path}' with HETERO:GPU.0,GPU.1 "
          f"pipeline-parallel policy...")
 
    model = core.read_model(model_xml_path)
 
    compiled_model = core.compile_model(
        model,
        device_name="HETERO:GPU.0,GPU.1",
        config={
            hints.model_distribution_policy: "PIPELINE_PARALLEL"
        }
    )
 
    print("Model compiled successfully.\n")
 
    infer_request = compiled_model.create_infer_request()
 
    # Costruisce input casuali coerenti con le shape attese dal modello,
    # così lo script funziona con qualunque modello IR senza doverlo
    # adattare manualmente ogni volta.
    inputs = {}
    for input_tensor in compiled_model.inputs:
        shape = input_tensor.get_shape()
        # Fallback per dimensioni dinamiche: usa 1 dove non specificato
        concrete_shape = [d if d > 0 else 1 for d in shape]
        dtype = input_tensor.get_element_type().to_dtype()
        inputs[input_tensor.get_any_name()] = np.random.rand(*concrete_shape).astype(dtype)
 
    successes = 0
    failures = 0
 
    for i in range(n_runs):
        print(f"--- Run {i + 1}/{n_runs} ---")
        try:
            result = infer_request.infer(inputs)
            print(f"  OK — output keys: {list(result.keys())[:3]}...")
            successes += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failures += 1
 
    print(f"\n=== Risultato finale: {successes}/{n_runs} successi, "
          f"{failures}/{n_runs} fallimenti ===")
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_hetero_low_level.py <path_al_modello.xml> [n_runs]")
        sys.exit(1)
 
    model_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
 
    main(model_path, runs)