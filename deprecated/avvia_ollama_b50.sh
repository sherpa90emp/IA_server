#!/bin/bash

source /opt/intel/oneapi/setvars.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate jarvis

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Parametri Ollama per Intel GPU
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=4096

# Gestione risorse
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export OLLAMA_INTEL_GPU=1

echo "----------------------------------------"
echo " Avvio Ollama su intel Arc Pro B50      "
echo " Ambiente: jarvis IPEX 2.8.0+xpu attivo " 
echo "----------------------------------------"

cd ~/llama-ccp
./ollama serve