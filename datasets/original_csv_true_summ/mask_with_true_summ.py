#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import ast


dataset_path='../mask_with_summ_dataset/'

if not os.path.exists(dataset_path):
	os.mkdir(dataset_path)

processd_path=os.path.join(dataset_path,'processed')
if not os.path.exists(processd_path):
	os.mkdir(processd_path)


# In[2]:


#load review data
review_df=pd.read_json('../hotel_mask_dataset/review.json')
#select necessary columns
business_df=review_df.sort_values('hotel_url')[['hotel_url']].drop_duplicates(['hotel_url'])
#executed om 26.9s


# In[7]:


# turn capital to un_capital letter
review_df.loc[:,['hotel_url']]=review_df.loc[:,['hotel_url']].applymap(lambda x:x.lower()).applymap(lambda x:x+'.json')
business_df.loc[:,['hotel_url']]=business_df.loc[:,['hotel_url']].applymap(lambda x:x.lower()).applymap(lambda x:x+'.json')


# In[10]:


#keep only intersection of diego's data 
hotel_with_summs_list=pd.read_csv('../true_summary/hotel_true_summs.csv')['hotel_url'].to_list()
index=review_df['hotel_url'].isin(hotel_with_summs_list)
review_df=review_df[index]
business_df=business_df[index]

#change the column name to text
review_df.rename(columns={'text_filtered_per_sentence':'text'},inplace=True)


# ### merge true_summary_df and business_df

# In[11]:


true_summary_df=pd.read_csv('../true_summary/hotel_true_summs.csv',index_col=0)


# In[12]:


true_summary_df


# In[13]:


#merge true_summary with business_df 
business_df=business_df.drop_duplicates(['hotel_url']).reset_index(drop=True)
business_df=pd.merge(business_df,true_summary_df,on='hotel_url',how='inner')


# In[14]:


business_df


# In[ ]:





# In[15]:


#output to json
review_df.to_json(os.path.join(dataset_path,'review.json'),orient='records')
business_df.to_json(os.path.join(dataset_path,'business.json'),orient='records')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




