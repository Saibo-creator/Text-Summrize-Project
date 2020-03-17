#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




df=pd.read_json('../hotel_mask_dataset/review_mask.json',orient='records')

# ## subword encoder



import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds




# Build
encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    df['text'].to_list(), target_vocab_size=32000)




import pickle
with open('../hotel_mask_dataset/processed/subwordenc_32000_secondpass.pkl', 'wb') as file:
    pickle.dump(encoder, file, protocol=pickle.HIGHEST_PROTOCOL)



