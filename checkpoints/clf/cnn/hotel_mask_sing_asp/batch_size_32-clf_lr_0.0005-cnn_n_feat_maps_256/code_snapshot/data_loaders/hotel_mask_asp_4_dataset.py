# hotel_dataset.py

"""
Data preparation and loaders for Hotel dataset. Here, an item is one "store" or "business"
"""
import random
import torch
from collections import Counter, defaultdict
import json
import math
import nltk
import numpy as np
import os
import pdb

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from data_loaders.summ_dataset import SummReviewDataset, SummDataset
from project_settings import HParams, DatasetConfig
from utils import load_file, save_file

from data_loaders.hotel_mask_dataset import Hotel_Mask_PytorchDataset, VariableNDocsSampler, Hotel_Mask_Dataset

class Hotel_Mask_Asp_PytorchDataset(Hotel_Mask_PytorchDataset):
    """
    Implements Pytorch Dataset

    One data point for model is n_reviews reviews for one item. When training, we want to have batch_size items and
    sample n_reviews reviews for each item. If a item has less than n_reviews reviews, we sample with replacement
    (sampling with replacement as then you'll be summarizing repeated reviews, but this shouldn't happen right now
    as only items with a minimum number of reviews is used (50). These items and their reviews are selected
    in YelpDataset.save_processed_splits().

    There is now also the option for variable n_docs -- see the documentations for n_reviews_min
    and n_reviews_max.
    """

    def __init__(self,
                 split=None,
                 n_reviews=None,
                 n_reviews_min=None,
                 n_reviews_max=None,
                 subset=None,
                 seed=0,
                 sample_reviews=True,
                 item_max_reviews=None):

        super(Hotel_Mask_Asp_PytorchDataset,self).__init__()
 

        self.ds_conf = DatasetConfig('hotel_mask_asp_4')  # used for paths



        self.items = self.load_all_items()

        # Create map from idx-th data point to item
        item_to_nreviews = load_file(
            os.path.join(self.ds_conf.processed_path, '{}/store-to-nreviews.json'.format(split)))
        self.idx_to_item = {}


        if sample_reviews:
            if n_reviews_min and n_reviews_max:
                self.idx_to_nreviews = {}
                self.idx_to_item_idxs = {}  # indices of reviews

                ns = [8] #[4, 8, 16]
                # ns = range(n_reviews_min, n_reviews_max+1, 4)  # e.g. [4,8,12,16]
                idx = 0
                for item, n_reviews in item_to_nreviews.items():
                    item_n = 0
                    selected_idxs = set()
                    while item_n < n_reviews:
                        # Keep selecting batches of reviews from this store (without replacement)
                        cur_n = random.choice(ns)
                        if item_n + cur_n > n_reviews:
                            break
                        available_idxs = set(range(n_reviews)).difference(selected_idxs)
                        cur_idxs = np.random.choice(list(available_idxs), cur_n, replace=False)
                        selected_idxs.update(cur_idxs)

                        # update
                        self.idx_to_item[idx] = item
                        self.idx_to_nreviews[idx] = cur_n
                        self.idx_to_item_idxs[idx] = cur_idxs
                        item_n += cur_n
                        idx += 1

            else:
                # Get the number of times each item will appear in a pass through this dataset
                item_min_reviews = min(item_to_nreviews.values())
                if item_max_reviews == float('inf'):
                    n_per_item = math.ceil(item_min_reviews / n_reviews)
                else:
                    n_per_item = np.mean([n for n in item_to_nreviews.values() if n <= item_max_reviews])
                    n_per_item = math.ceil(n_per_item / n_reviews)
                # print('Each item will appear {} times'.format(n_per_item))

                idx = 0
                for item, n_reviews in item_to_nreviews.items():
                    if n_reviews <= item_max_reviews:
                        for _ in range(n_per_item):
                            self.idx_to_item[idx] = item
                            idx += 1
        else:
            # __getitem__ will not sample
            idx = 0
            self.idx_to_item_startidx = {}
            # idx items idx of one dataset item. item_startidx is the idx within that item's reviews.
            tot = 0
            for item, item_n_reviews in item_to_nreviews.items():
                if item_n_reviews <= item_max_reviews:
                    tot += item_n_reviews
                    item_startidx = 0
                    for _ in range(math.floor(item_n_reviews / n_reviews)):
                        self.idx_to_item[idx] = item
                        self.idx_to_item_startidx[idx] = item_startidx
                        idx += 1
                        item_startidx += n_reviews

        if self.subset:
            end = int(self.subset * len(self.idx_to_item))
            for idx in range(end, len(self.idx_to_item)):
                del self.idx_to_item[idx]

        self.n = len(self.idx_to_item)


class Hotel_Mask_Asp_4_Dataset(Hotel_Mask_Dataset):
    """
    Main class for using Hotel dataset
    """
    def __init__(self):
        super(Hotel_Mask_Asp_4_Dataset, self).__init__()
        self.name = 'hotel_mask_asp_4'
        self.conf = DatasetConfig('hotel_mask_asp_4')
        self.subwordenc = load_file(self.conf.subwordenc_path)


if __name__ == '__main__':
    from data_loaders.summ_dataset_factory import SummDatasetFactory

    hp = HParams()
    ds = SummDatasetFactory.get('hotel_mask_asp_4')
    ds.save_processed_splits()
   
