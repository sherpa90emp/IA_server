#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate jarvis

export LD_LIBRARY_PATH=opt/intel/oneapi/dnnl/2025.3/lib:$LD_LIBRARY_PATH

echo "--------------------------------------------------"
echo " Client Ollama pronto                             "
echo " lancia comando es: ./ollama run qwen2.5-coder:7b "
echo "--------------------------------------------------"

cd ~/llama-ccp
