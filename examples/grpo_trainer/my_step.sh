#!/bin/bash
set -e
conda activate zzn
ray stop
ray stop --force
ray start --head --port=6379 --dashboard-port=8265 --num-gpus=2
cd verl/examples/grpo_trainer
bash my.sh