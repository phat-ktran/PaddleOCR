#!/bin/bash

# Navigate to the workspace directory
cd /workspace/PaddleOCR/

# Hugging Face dataset
huggingface-cli download benvijamin/rec_paddle_v7_10m \
  --repo-type dataset \
  --local-dir ./train_data/rec \
  --token $HF_KEY

huggingface-cli download benvijamin/rec_paddle_v7.1_eval \
    --repo-type dataset \
    --local-dir ./train_data/rec \
    --token $HF_KEY

wandb login $WB_KEY
