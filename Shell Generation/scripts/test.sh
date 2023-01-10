#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:59
source /home/golriz/projects/def-guibault/golriz/env/bin/activate
set -x
GPUS=$1
PY_ARGS=${@:2}
CUDA_VISIBLE_DEVICES=${GPUS} python main.py --test  ${PY_ARGS}}
