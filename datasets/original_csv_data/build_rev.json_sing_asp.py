#!/usr/bin/env python
# coding: utf-8

# In[1]:

import ast
import pandas as pd
import numpy as np



# In[24]:


test=pd.read_csv('./test.csv')
train=pd.read_csv('./train.csv')
dev=pd.read_csv('./dev.csv')
'''executed in 2m 49s'''


# In[3]:


len(train),len(test),len(dev)


# In[4]:



def parse_str_list(df):
    df['text_filtered_per_sentence']=df['text_filtered_per_sentence'].apply(ast.literal_eval)
    df['aspect_filtered_per_sentence']=df['aspect_filtered_per_sentence'].apply(ast.literal_eval)
    
    
parse_str_list(test)
parse_str_list(train)
parse_str_list(dev)

'''executed in 2m 24s'''



# ## check data

# In[11]:


test1=pd.read_csv('./test.csv')


# In[12]:


test1.head(3).iloc[0]['property_dict']


# In[13]:


def groupByAspect(split):
    split_sentence=split[['hotel_url','text','text_filtered_per_sentence']].explode('text_filtered_per_sentence')
    split_aspect=split[['hotel_url','text','aspect_filtered_per_sentence']].explode('aspect_filtered_per_sentence')
    split_sentence.reset_index(drop=True,inplace=True)
    split_aspect.reset_index(drop=True,inplace=True)
    split_exploded=split_aspect.join(split_sentence[['text_filtered_per_sentence']])
    
    grouped=pd.DataFrame(split_exploded.groupby(['hotel_url','aspect_filtered_per_sentence'])['text_filtered_per_sentence'].apply(list))
    grouped.reset_index(inplace=True)
    grouped['text_filtered_per_sentence']=grouped['text_filtered_per_sentence'].apply(lambda data: [data[x:x+5] for x in range(0, len(data), 5)])
    grouped.head()
    
    grouped=grouped.explode('text_filtered_per_sentence').reset_index()
    grouped['text']=grouped['text_filtered_per_sentence'].apply(lambda x:''.join(x))
    grouped=grouped.rename(columns={'hotel_url':'hotel_url_original'})
    grouped['hotel_url']=grouped['hotel_url_original']
    grouped=grouped[['index', 'hotel_url', 'aspect_filtered_per_sentence',
           'text_filtered_per_sentence', 'text','hotel_url_original']]
    
    grouped['hotel_url']='aspect '+grouped['aspect_filtered_per_sentence'].apply(str)+' '+grouped['hotel_url_original']
    return grouped


# In[14]:


"executed in 14m 6s"

test=groupByAspect(test)
train=groupByAspect(train)
dev=groupByAspect(dev)
test.head()


# In[16]:


test['rating']=-1
train['rating']=-1
dev['rating']=-1


# In[27]:


test.head()


# In[60]:


aspect_rating=test.replace(-1,np.nan).groupby('hotel_url')[["service","cleanliness","value","location","rooms"]].mean()


# In[62]:


aspect_rating=aspect_rating.rename(columns={'service':'aspect 0 ','cleanliness':'aspect 1 ',                              'value':'aspect 2 ','location':'aspect 3 ','rooms':'aspect 4 '}).reset_index()


# In[63]:


aspect_rating=pd.melt(aspect_rating, id_vars='hotel_url', value_vars=['aspect 0 ','aspect 1 ','aspect 2 '                                                          ,'aspect 3 ','aspect 4 ',])


# In[64]:


aspect_rating['hotel_url_new']=aspect_rating['variable']+aspect_rating['hotel_url']
aspect_rating=aspect_rating[['hotel_url_new','value']].rename(columns={'hotel_url_new':'hotel_url'})


# In[67]:


aspect_rating


# #### raw code

# In[10]:


test_sentence=test[['hotel_url','text','text_filtered_per_sentence']].explode('text_filtered_per_sentence')
test_aspect=test[['hotel_url','text','aspect_filtered_per_sentence']].explode('aspect_filtered_per_sentence')


# In[11]:


len(test_sentence),len(test_aspect)


# In[12]:


test_sentence.reset_index(drop=True,inplace=True)
test_aspect.reset_index(drop=True,inplace=True)


# In[13]:


test_exploded=test_aspect.join(test_sentence[['text_filtered_per_sentence']])


# In[14]:


test_exploded.head()


# In[15]:


# '''3min53s/1min 49s/17s'''
# grouped=pd.DataFrame(test_exploded.groupby(['hotel_url','aspect_filtered_per_sentence'])['text_filtered_per_sentence'].apply(lambda x: ','.join(x)))
grouped=pd.DataFrame(test_exploded.groupby(['hotel_url','aspect_filtered_per_sentence'])['text_filtered_per_sentence'].apply(list))
grouped.reset_index(inplace=True)


# In[ ]:





# In[16]:


"""executed in 5.53s """
grouped['text_filtered_per_sentence']=grouped['text_filtered_per_sentence'].apply(lambda data: [data[x:x+5] for x in range(0, len(data), 5)])
grouped.head()


# In[19]:


grouped=grouped.explode('text_filtered_per_sentence').reset_index()


# In[20]:


grouped.head()


# In[22]:


grouped['text']=grouped['text_filtered_per_sentence'].apply(lambda x:''.join(x))


# In[28]:


grouped.head()


# In[29]:


grouped=grouped.rename(columns={'hotel_url':'hotel_url_original'})
grouped['hotel_url']=grouped['hotel_url_original']
grouped=grouped[['index', 'hotel_url', 'aspect_filtered_per_sentence',
       'text_filtered_per_sentence', 'text','hotel_url_original']]


# In[42]:


grouped['hotel_url']='aspect '+grouped['aspect_filtered_per_sentence'].apply(str)+' '+grouped['hotel_url_original']


# In[44]:


grouped.head(10)


# In[21]:


grouped['text_filtered_per_sentence'].apply(len).describe()




# ## to_json

# In[18]:


"executed in 51.6s"
tot=pd.concat([dev,test,train])


# In[19]:


"executed in 1m 1.82s"
df=tot.sort_values('hotel_url')[['hotel_url','text','rating']]
#df=tot.sort_values('hotel_url')[['hotel_url','rating']]


# In[20]:


"executed in 30.0"
dataset_path='../hotel_mask_sing_asp_dataset/'
df.to_json(os.path.join(dataset_path,'review.json'),orient='records')

src=os.path.join(dataset_path,'review.json')
dst=os.path.join(dataset_path,'business.json')
copyfile(src, dst)


