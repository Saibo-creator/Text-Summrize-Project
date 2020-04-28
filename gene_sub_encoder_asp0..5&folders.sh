#!/usr/bin/env bash


cur=`pwd`
echo $cur

for f in hotel_mask_asp_0 hotel_mask_asp_1 hotel_mask_asp_2 hotel_mask_asp_3 hotel_mask_asp_4
do
mkdir -p "datasets/${f}_dataset/processed/subwordenc/"

mkdir -p "checkpoints/lm/mlstm/${f}"
mkdir -p "checkpoints/clf/cnn/${f}"
mkdir -p "checkpoints/sum/mlstm/${f}"
mkdir -p "outputs/eval/${f}"

PYTHONPATH=.  python3 data_loaders/build_subword_encoder.py --dataset=$f\
--output_dir="./datasets/${f}_dataset/processed/subwordenc/" --output_fn=subwordenc


done





# python3 data_loaders/build_subword_encoder.py --dataset=hotel_mask_sing_asp \
# --output_dir=./datasets/hotel_mask_sing_asp_dataset/processed/subwordenc/ --output_fn=subwordenc



# python3 data_loaders/build_subword_encoder.py --dataset=hotel_mask_sing_asp \
# --output_dir=./datasets/hotel_mask_sing_asp_dataset/processed/subwordenc/ --output_fn=subwordenc


