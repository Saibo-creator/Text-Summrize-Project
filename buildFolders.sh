#!/usr/bin/env bash



for f in hotel_mask_asp_0 hotel_mask_asp_1 hotel_mask_asp_2 hotel_mask_asp_3 hotel_mask_asp_4
do
mkdir -p "datasets/${f}_dataset/processed/subwordenc/"

mkdir -p "checkpoints/lm/mlstm/${f}"
mkdir -p "checkpoints/clf/cnn/${f}"
mkdir -p "checkpoints/sum/train/${f}"
mkdir -p "checkpoints/sum/mlstm/${f}"
mkdir -p "outputs/eval/${f}"

# PYTHONPATH=.  python3 data_loaders/build_subword_encoder.py --dataset=$f\
# --output_dir="./datasets/${f}_dataset/processed/subwordenc/" --output_fn=subwordenc


done
