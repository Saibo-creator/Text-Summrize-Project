# summ_dataset_factory.py

"""
Class to create SummDataset instances.

In part here in a separate file to avoid circular imports
"""

from data_loaders.amazon_dataset import AmazonDataset
from data_loaders.yelp_dataset import YelpDataset
from data_loaders.hotel_dataset import HotelDataset
from data_loaders.hotel_mask_dataset import Hotel_Mask_Dataset


class SummDatasetFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get(name):
        if name == 'amazon':
            return AmazonDataset()
        elif name == 'yelp':
            return YelpDataset()
        elif name == 'hotel':
            return HotelDataset()
        elif name == 'mask_hotel':
            return Hotel_Mask_Dataset()
