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



class Hotel_Mask_Asp_PytorchDataset(Dataset):
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
        """
        Args:
            split: str ('train', val', 'test')
            n_reviews: int

            n_reviews_min: int
            n_reviews_max: int
                - When these two are provided, then there will be variable n_reviews (i.e. two different
                training examples may be composed of different number of reviews to summarize)
                - Some of this

            subset: float (Value in [0.0, 1.0]. If given, then dataset is truncated to subset of the businesses
            seed: int (set seed because we will be using np.random.choice to sample reviews if sample_reviews=True)
            sample_reviews: boolean
                - When True, __getitem_ will sample n_reviews reviews for each item. The number of times a item appears
                in the dataset is dependent on uniform_items.
                - When False, each item will appear math.floor(number of reviews item has / n_reviews) times
                so that almost every review is seen (with up to n_reviews - 1 reviews not seen).
                    - Setting False is useful for (a) validation / test, and (b) simply iterating over all the reviews
                    (e.g. to build the vocabulary).
            item_max_reviews: int (maximum number of reviews a item can have)
                - This is used to remove outliers from the data. This is especially important if uniform_items=False,
                as there may be a large number of reviews in a training epoch coming from a single item. This also
                still matters when uniform_items=True, as items an outlier number of reviews will have reviews
                that are never sampled.
                
                
                - For the Hotel dataset, there are:
                 257,186 reviews in the test set
                 1,749,158 reviews in the train set
                 216,031 reviews in the dev set
                 
                - For the Yelp dataset
                 6,685,900 reviews in all
                 192,609 business
                
        """
        self.split = split

        self.n_reviews = n_reviews
        self.n_reviews_min = n_reviews_min
        self.n_reviews_max = n_reviews_max

        self.subset = subset
        self.sample_reviews = sample_reviews
        item_max_reviews = float('inf') if item_max_reviews is None else item_max_reviews
        self.item_max_reviews = item_max_reviews

        self.ds_conf = DatasetConfig('hotel_mask_asp_1')  # used for paths

        # Set random seed so that choice is always the same across experiments
        # Especially necessary for test set (along with shuffle=False in the DataLoader)
        np.random.seed(seed)

        self.items = self.load_all_items()

        # Create map from idx-th data point to item
        item_to_nreviews = load_file(
            os.path.join(self.ds_conf.processed_path, '{}/store-to-nreviews.json'.format(split)))
        self.idx_to_item = {}

        ''' item_to_nreviews=store-to-nreviews.json
          id : number of review on this hotel/store
          "z8XIldYr65CKOz8yOI5R8A": 217,
          "zCmwErM82hdKl4L2UixDJw": 309,
          "zDKH2ua2DauqiWOAP9yz7g": 205,
          "zE7pCmT1P9-tFLHcoJ7jQw": 50,
          "zEaGcSVPDQipnRdEZp-F6g": 413,
          "zHYjts6JqKBiKM5ZbO8eHA": 78,
          "zINKP9rtvo2IUKTZoYJUvQ": 151,
          "zJGtD3y-pAIGNId4codEEg": 539,
          "zTNFyAXMEkgVP-o_HTC2xw": 143,
          "zcZn9_qCLqRU8qsxYXtOBQ": 78,
          "zdE82PiD6wquvjYLyhOJNA": 867,
          "zfMA___zTTBGzZPsKZFWxw": 51,
          "zjwdU1OdlbKTGjm-IfD4TQ": 118,
          "zkhBU5qW_zCy0q4OEtIrsA": 217,
          "zm-nB9xWL0RWZ-zoL7JNuQ": 76,
          "znWHLW1pt19HzW1VY6KfCA": 468,
          "zr42_UsWfaIF-rcp37OpwA": 144,
          "zrTGcb83AsfyVTMrsCa65A": 352,
          "zsYYAnHEs5qzsfazSsGkBw": 94,
          "zwvshlu1bE2na9sXYrP0TQ": 81







        '''
        if sample_reviews:
            if n_reviews_min and n_reviews_max:
                self.idx_to_nreviews = {}
                self.idx_to_item_idxs = {}  # indices of reviews

                ns = [8]#[4, 8, 16]
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

    def load_all_items(self):
        """
        Return dictionary from item id to dict
        """
        print('Loading all items')
        items = {}
        with open(self.ds_conf.businesses_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)[0]
                items[line['hotel_url']] = line # key = hotel_url , value=  "name": "Bloom",
                                                                  # "address": "8877 N Scottsdale Rd, Ste 402",
                                                                  # "city": "Scottsdale",
                                                                  # "state": "AZ",
                                                                  # "postal_code": "85253",
                                                                  # "latitude": 33.5663822,
                                                                  # "longitude": -111.925581,

        return items

    def __getitem__(self, idx):
        # Map idx to item and load reviews
        item = self.idx_to_item[idx]  # id
        fp = os.path.join(self.ds_conf.processed_path, '{}/{}_reviews.json'.format(self.split, item))
        reviews = load_file(fp)

        # Get reviews from item
        if self.sample_reviews:
            if self.n_reviews_min and self.n_reviews_max:
                review_idxs = self.idx_to_item_idxs[idx]
                reviews = [reviews[r_idx] for r_idx in review_idxs]
            else:
                if len(reviews) < self.n_reviews:
                    reviews = np.random.choice(reviews, size=self.n_reviews, replace=True)
                else:
                    reviews = np.random.choice(reviews, size=self.n_reviews, replace=False)
        else:
            start_idx = self.idx_to_item_startidx[idx]
            reviews = reviews[start_idx:start_idx + self.n_reviews]

        # Collect data for this item
        hotel_ids,texts, ratings = zip(*[(s['hotel_url'],s['text'], s['rating']) for s in reviews])
        texts = SummDataset.concat_docs(texts, edok_token=True)
        avg_rating = int(np.round(np.mean(ratings)))
        hotel_id=hotel_ids[0]

        try:
            categories = '---'.join(self.items[item]['categories'])
        except Exception as e:
            print(e)
            categories = '---'
        metadata={'item': item,}

        print('avg_rating',avg_rating)
        return hotel_id, texts, avg_rating, metadata

    def __len__(self):
        return self.n

class VariableNDocsSampler(Sampler):
    """
    Produce indices for variable n_docs at a time. Used in conjunction with
    n_docs_min and n_docs_max, which creates the dictionaries needed in
    YelpPytorchDataset.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset):
        super(VariableNDocsSampler, self).__init__(dataset)
        self.dataset = dataset

        # Group data points together by how the number of reviews
        # This way the SummarizationModel will be fed a batch of points, each summarizing
        # the same number of reviews. This is important as the model reshapes tensors
        # by n_docs, which allows it to be done in parallel.
        nreviews_to_idxs = defaultdict(list)
        for idx, nreviews in dataset.idx_to_nreviews.items():
            nreviews_to_idxs[nreviews].append(idx)

        # This is hard-coded: the summarization model I've been training takes about
        # 10 GB on one GPU for batch_size = 4 and n_docs = 8. We'll scale the batch_size
        # relative to the n_docs for the given minibatch such that
        # batch_size * n_docs  / ngpus = 32, so as to use all the GPU memory as much
        # as possible. For n_docs_min=4 and n_docs_max=16, n_docs is in [4,8,12,16]
        # (this is hard-coded in currently in YelpPytorchDataset).
        ngpus = torch.cuda.device_count()

        dataloader_idxs = []  # list of lists of indices, each sublist is a minibatch
        for nreviews, idxs in nreviews_to_idxs.items():
            batch_size = int(32 / nreviews * ngpus)
            print(nreviews, batch_size)
            selected_idxs = set()
            while len(set(idxs).difference(selected_idxs)) > batch_size:
                # There is enough unselected points to form a batch
                available_idxs = set(idxs).difference(selected_idxs)
                cur_idxs = np.random.choice(list(available_idxs), batch_size, replace=False)
                dataloader_idxs.append(cur_idxs)
                selected_idxs.update(cur_idxs)

        random.shuffle(dataloader_idxs)

        self.dataloader_idxs = dataloader_idxs

    def __iter__(self):
        return iter(self.dataloader_idxs)

    def __len__(self):
        return len(self.dataloader_idxs)




class Hotel_Mask_Asp_1_Dataset(SummReviewDataset):
    """
    Main class for using Hotel dataset
    """
    def __init__(self):
        super(Hotel_Mask_Asp_1_Dataset, self).__init__()
        self.name = 'hotel_mask_asp_1'
        self.conf = DatasetConfig('hotel_mask_asp_1')
        self.n_ratings_labels = 5
        self.reviews = None
        self.subwordenc = load_file(self.conf.subwordenc_path)





    ####################################
    #
    # Utils
    #
    ####################################


####### below done  #####
    def load_all_reviews(self):
        """
        Return list of dictionaries
        """
        print('Loading all reviews')
        reviews = []
        with open(self.conf.reviews_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                reviews.append(json.loads(line))
        return reviews
####### above done  #####



    def get_data_loader(self, split='train',
                        n_docs=8, n_docs_min=None, n_docs_max=None,
                        subset=None, seed=0, sample_reviews=True,
                        category=None,  # for compatability with AmazonDataset, which filters in AmazonPytorchDataset
                        batch_size=64, shuffle=True, num_workers=4):
        """
        Return iterator over specific split in dataset
        """
        ds = Hotel_Mask_Asp_PytorchDataset(split=split,
                                n_reviews=n_docs, n_reviews_min=n_docs_min, n_reviews_max=n_docs_max,
                                subset=subset, seed=seed, sample_reviews=sample_reviews,
                                item_max_reviews=self.conf.item_max_reviews)

        if n_docs_min and n_docs_max:
            loader = DataLoader(ds, batch_sampler=VariableNDocsSampler(ds), num_workers=num_workers)
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

    ####################################
    #
    # One off functions
    #
    ####################################
    def save_processed_splits(self):
        """
        Save train, val, and test splits. Splits are across items (e.g. a item is either in train, val, or test).
        Iterates over all reviews in the original dataset. Tries to get close to a 80-10-10 split.

        Args:
            review_max_len: int (maximum length in subtokens a review can be)
            item_min_reviews: int (min number of reviews a item must have)
        """
        review_max_len = self.conf.review_max_len
        item_min_reviews = self.conf.item_min_reviews

        print('Saving processed splits')
        if self.reviews is None:
            self.reviews = self.load_all_reviews()

        print('Filtering reviews longer than: {}'.format(review_max_len))
        item_to_reviews = defaultdict(list)
        
        for r in self.reviews[0]:
            if len(self.subwordenc.encode(r['text'])) < review_max_len:
                item_to_reviews[r['hotel_url']].append(r)

        # Calculate target amount of reviews per item
        n = sum([len(revs) for revs in item_to_reviews.values()])
        print('Total number of reviews before filtering: {}'.format(len(self.reviews[0])))
        print('Total number of reviews after filtering: {}'.format(n))

        print('Filtering items with less than {} reviews'.format(item_min_reviews))
        item_to_n = {}
        for item in list(item_to_reviews.keys()):  # have to do list and keys for python3 to delete in-place
            # for item, reviews in item_to_reviews.items():
            n = len(item_to_reviews[item])
            if n < item_min_reviews:
                del item_to_reviews[item]
            else:
                item_to_n[item] = n
        n = sum(item_to_n.values())
        print('Total number of reviews after filtering: {}'.format(n))
        print('Total number of items after filtering: {}'.format(len(item_to_n)))

        # Construct splits
        n_tr, n_val, n_te = int(0.8 * n), int(0.1 * n), int(0.1 * n)
        cur_n_tr, cur_n_val, cur_n_te = 0, 0, 0
        split_to_item_to_nreviews = {'train': {}, 'val': {}, 'test': {}}
        # In descending order of number of reviews per item
        for i, (item, n) in enumerate(sorted(item_to_n.items(), key=lambda x: -x[1])):
            # once every ten items, save to val / test if we haven't yet hit the target number
            if (i % 10 == 8) and (cur_n_val < n_val):
                split = 'val'
                cur_n_val += n
            elif (i % 10 == 9) and (cur_n_te < n_te):
                split = 'test'
                cur_n_te += n
            else:
                split = 'train'
                cur_n_tr += n

            out_fp = os.path.join(self.conf.processed_path, '{}/{}_reviews.json'.format(split, item))
            save_file(item_to_reviews[item], out_fp, verbose=False)

            split_to_item_to_nreviews[split][item] = n

        print('Number of train reviews: {} / {}'.format(cur_n_tr, n_tr))
        print('Number of val reviews: {} / {}'.format(cur_n_val, n_val))
        print('Number of test reviews: {} / {}'.format(cur_n_te, n_te))

        # This file is used by YelpPytorchDataset
        for split, item_to_nreviews in split_to_item_to_nreviews.items():
            out_fp = os.path.join(self.conf.processed_path, '{}/store-to-nreviews.json'.format(split))
            save_file(item_to_nreviews, out_fp)



if __name__ == '__main__':
    from data_loaders.summ_dataset_factory import SummDatasetFactory

    hp = HParams()
    ds = SummDatasetFactory.get('hotel_mask_asp_1')
    ds.save_processed_splits()
   
