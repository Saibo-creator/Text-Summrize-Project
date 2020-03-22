import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import ast

dataset_path='../hotel_mask_dataset/'

if not os.path.exists(dataset_path):
	os.mkdir(dataset_path)

processd_path=os.path.join(dataset_path,'processed')
if not os.path.exists(processd_path):
	os.mkdir(processd_path)

test=pd.read_csv('./test.csv')
train=pd.read_csv('./train.csv')
dev=pd.read_csv('./dev.csv')


tot=pd.concat([dev,test,train])
tot_filtered=tot.copy()
tot_filtered['text_filtered_per_sentence']=tot_filtered['text_filtered_per_sentence'].apply(ast.literal_eval).apply(lambda x:"".join(x) )


df=tot_filtered.sort_values('hotel_url')[['hotel_url','text_filtered_per_sentence','rating']]
df.to_json(os.path.join(dataset_path,'review_mask.json'),orient='records')

src=os.path.join(dataset_path,'review_mask.json')
dst=os.path.join(dataset_path,'business_mask.json')
copyfile(src, dst)


