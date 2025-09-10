#!/bin/bash

export WANDB_MODE=offline   # disable online logging
export FLAGS_allocator_strategy='auto_growth'
export FLAGS_fraction_of_gpu_memory_to_use=0.98
# Number of parallel jobs (adjust as needed)
JOBS=2

# Total runs
N=12

parallel -j $JOBS --lb "
python tools/eval.py \
  -c configs/rec/PP-Thesis/Nom/main/b2/pdnom/rec_svtrnet_base_nom_exp_b2.1.1.yml \
  -o Global.checkpoints=./output/rec_svtrnet_base_nom_exp_b2.1.1/best_accuracy \
     Eval.dataset.data_dir=./train_data/ \
     Eval.dataset.label_file_list=\"['./train_data/Labels/Validate/nomna_validate_old.txt']\" \
     Eval.loader.batch_size_per_card=12 \
     Global.save_res_path=./local/dropout/drop_{#}.txt \
     Global.filename_idx=3 \
     Architecture.Backbone.enable_dropout=True
" ::: $(seq 1 $N)
