# summ_dataset_factory.py

"""
Class to create SummDataset instances.

In part here in a separate file to avoid circular imports
"""

from data_loaders.amazon_dataset import AmazonDataset
from data_loaders.yelp_dataset import YelpDataset
from data_loaders.hotel_dataset import HotelDataset
from data_loaders.hotel_mask_dataset import Hotel_Mask_Dataset
from data_loaders.hotel_mask_sing_asp_dataset import Hotel_Mask_Sing_Asp_Dataset
from data_loaders.hotel_mask_asp_0_dataset import Hotel_Mask_Asp_0_Dataset
from data_loaders.hotel_mask_asp_1_dataset import Hotel_Mask_Asp_1_Dataset
from data_loaders.hotel_mask_asp_2_dataset import Hotel_Mask_Asp_2_Dataset
from data_loaders.hotel_mask_asp_3_dataset import Hotel_Mask_Asp_3_Dataset
from data_loaders.hotel_mask_asp_4_dataset import Hotel_Mask_Asp_4_Dataset


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
        elif name == 'hotel_mask':
            return Hotel_Mask_Dataset()
        elif name == 'hotel_mask_sing_asp':
            return Hotel_Mask_Sing_Asp_Dataset()
        elif name == 'hotel_mask_asp_0':
            return Hotel_Mask_Sing_Asp_Dataset()
        elif name == 'hotel_mask_asp_1':
            return Hotel_Mask_Sing_Asp_Dataset()
        elif name == 'hotel_mask_asp_2':
            return Hotel_Mask_Sing_Asp_Dataset()
        elif name == 'hotel_mask_asp_3':
            return Hotel_Mask_Sing_Asp_Dataset()
        elif name == 'hotel_mask_asp_4':
            return Hotel_Mask_Sing_Asp_Dataset()


