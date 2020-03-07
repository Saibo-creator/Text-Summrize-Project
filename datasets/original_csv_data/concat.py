#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




test=pd.read_csv('./test.csv')
train=pd.read_csv('./train.csv')
dev=pd.read_csv('./dev.csv')

tot=pd.concat([dev,test,train])
df=tot.sort_values('hotel_url')[['hotel_url','text','rating']]
df.to_json('../review.json',orient='records')
