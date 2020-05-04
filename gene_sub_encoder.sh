#!/usr/bin/env bash



for f in hotel_mask_asp_0 hotel_mask_asp_1 hotel_mask_asp_2 hotel_mask_asp_3 hotel_mask_asp_4
do

mkdir -p "datasets/${f}_dataset/processed/subwordenc/"

mkdir -p "checkpoints/lm/mlstm/${f}"
mkdir -p "checkpoints/clf/cnn/${f}"
mkdir -p "checkpoints/sum/train/${f}"
mkdir -p "checkpoints/sum/mlstm/${f}"
mkdir -p "outputs/eval/${f}"

bash scripts/preprocess_data.sh "${f}_dataset.py"

# PYTHONPATH=.  python3 data_loaders/build_subword_encoder.py --dataset=${f} \
# --output_dir="./datasets/${f}_dataset/processed/subwordenc/" --output_fn=subwordenc


done



# r"""Program to build a SubwordTextEncoder.

# Example usage:

# PYTHONPATH=. python data_loaders/build_subword_encoder.py \
# --dataset=yelp \
# --output_dir=./ \
# --output_fn=subwordenc

# PYTHONPATH=. python data_loaders/build_subword_encoder.py \
# --corpus_filepattern=datasets/yelp_dataset/processed/reviews_texts_train.txt \
# --output_dir=./ \
# --output_fn=tmp_enc


# PYTHONPATH=. python data_loaders/build_subword_encoder.py \
# --dataset=hotel_mask \
# --output_dir=./ \
# --output_fn=subwordenc


# PYTHONPATH=. python data_loaders/build_subword_encoder.py \
# --dataset=hotel_mask_asp_1 \
# --output_dir=./ \
# --output_fn=subwordenc


# """



