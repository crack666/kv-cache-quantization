#!/bin/bash
cd "/mnt/a/Users/crack.crackdesk/Documents/Beuth it up/Master 3/Wissenschaftliches Projekt/kv-cache-quantization"
source /home/crack/.venvs/global/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
python3 scripts/experiment_kv_split.py "$@"
