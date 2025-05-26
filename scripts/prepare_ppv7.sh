#!/bin/bash

# Navigate to the workspace directory
cd /workspace/PaddleOCR/

# Hugging Face dataset
huggingface-cli download benvijamin/rec_paddle_v7_part1 \
  --repo-type dataset \
  --local-dir ./train_data/rec \
  --token $HF_KEY

huggingface-cli download benvijamin/rec_paddle_v7_part2 \
    --repo-type dataset \
    --local-dir ./train_data/rec \
    --token $HF_KEY

mkdir ./train_data/rec/train
mkdir ./train_data/rec/val

# Move data to fit with configuration
mv  ./train_data/rec/rec_paddle_v7_part1/train/* ./train_data/rec/train
mv  ./train_data/rec/rec_paddle_v7_part1/val/* ./train_data/rec/val
mv  ./train_data/rec/rec_paddle_v7_part2/train/* ./train_data/rec/train
mv  ./train_data/rec/rec_paddle_v7_part2/val/* ./train_data/rec/val

wandb login $WB_KEY
