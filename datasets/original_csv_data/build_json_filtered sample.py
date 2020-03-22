import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import ast



test=pd.read_csv('./sample_test.csv')
# train=pd.read_csv('./train.csv')
# dev=pd.read_csv('./dev.csv')


tot=pd.concat([test])
tot_filtered=tot.copy()
tot_filtered['text_filtered_per_sentence']=tot_filtered['text_filtered_per_sentence'].apply(ast.literal_eval).apply(lambda x:"".join(x) )


df=tot_filtered.sort_values('hotel_url')[['hotel_url','text_filtered_per_sentence','rating']]
df.to_json('../hotel_mask_dataset/review_mask.json',orient='records')

src='../hotel_mask_dataset/review.json'
dst='../hotel_mask_dataset/business.json'
copyfile(src, dst)


