#!/bin/bash

# Navigate to the workspace directory
cd /workspace/PaddleOCR/

# Hugging Face dataset
huggingface-cli download chulanpro5/datasets \
  --include "hisdoc1b_greedy_41k/*" \
  --repo-type dataset \
  --local-dir ./train_data/rec \
  --token $TRAIN_HF_KEY
  
mv ./train_data/rec/hisdoc1b_greedy_41k/* ./train_data/rec/

huggingface-cli download benvijamin/rec_paddle_v7.1_eval \
    --repo-type dataset \
    --local-dir ./train_data/rec \
    --token $EVAL_HF_KEY

wandb login $WB_KEY
