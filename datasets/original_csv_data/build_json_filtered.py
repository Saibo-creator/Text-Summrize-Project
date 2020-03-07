import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import ast



test=pd.read_csv('./test.csv')
train=pd.read_csv('./train.csv')
dev=pd.read_csv('./dev.csv')


tot=pd.concat([dev,test,train])
tot_filtered=tot.copy()
tot_filtered['text_filtered_per_relation_grouped']=tot_filtered['text_filtered_per_relation'].apply(ast.literal_eval).apply(set).apply(lambda x:"\\".join(x) )


df=tot_filtered.sort_values('hotel_url')[['hotel_url','text_filtered_per_relation_grouped','rating']]
df.to_json('../hotel_mask_dataset/review_filtered.json',orient='records')

src='../hotel_mask_dataset/review_filtered.json'
dst='../hotel_mask_dataset/business_filtered.json'
copyfile(src, dst)


