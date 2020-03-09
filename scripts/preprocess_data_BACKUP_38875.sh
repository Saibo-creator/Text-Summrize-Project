#!/usr/bin/env bash

# Can execute script from anywhere
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ..

#mv datasets/yelp_dataset/yelp_academic_dataset_review.json datasets/yelp_dataset/review.json  (out of date)
# PYTHONPATH=. python data_loaders/yelp_dataset.py

#PYTHONPATH=. python3 data_loaders/hotel_dataset.py
<<<<<<< HEAD

=======
>>>>>>> b88918f7823c3056b6ccf6a7c2cd841f14098ce5
PYTHONPATH=. python3 data_loaders/hotel_mask_dataset.py
