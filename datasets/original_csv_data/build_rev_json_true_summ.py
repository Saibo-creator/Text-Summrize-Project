import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile
import ast

dataset_path='../hotel_with_summ_dataset/'

if not os.path.exists(dataset_path):
	os.mkdir(dataset_path)

processd_path=os.path.join(dataset_path,'processed')
if not os.path.exists(processd_path):
	os.mkdir(processd_path)

# test=pd.read_csv('./test.csv').head(1000)
# train=pd.read_csv('./train.csv').head(1000)
dev=pd.read_csv('./dev.csv').head(1000)


# tot=pd.concat([dev,test,train])
tot=pd.concat([dev])

#select necessary columns
review_df=tot.sort_values('hotel_url')[['hotel_url','text','rating']]
business_df=tot.sort_values('hotel_url')[['hotel_url']]


# turn capital to un_capital letter
review_df.loc[:,['hotel_url']]=review_df.loc[:,['hotel_url']].applymap(lambda x:x.lower())
business_df.loc[:,['hotel_url']]=business_df.loc[:,['hotel_url']].applymap(lambda x:x.lower())

#keep only intersection of diego's data 
hotel_with_summs_list=pd.read_csv('../true_summary/hotel_true_summs.csv')['hotel_url'].to_list()
index=review_df['hotel_url'].isin(hotel_with_summs_list)
review_df=review_df[index]
business_df=business_df[index]

#output to json
review_df.to_json(os.path.join(dataset_path,'review.json'),orient='records')
business_df.to_json(os.path.join(dataset_path,'business.json'),orient='records')
#output to csv
review_df.to_csv(os.path.join(dataset_path,'review.csv'))
business_df.to_csv(os.path.join(dataset_path,'business.csv'))


#original hotel name"Hotel_Review-g28970-d84032-Reviews-The_Graham_Washington_DC_Georgetown_Tapestry_Collection_by_Hilton-Washington_DC_District_review.html.json"
#=>Hotel_Review-g29196-d89254-Reviews-Homewood_Suites_by_Hilton_Atlanta_Alpharetta-Alpharetta_Georgia.html.json

#Diego:"hotel_review-g28970-d84029-reviews-the_georgetown_inn-washington_dc_district_of_columbia.json"
#=>hotel_review-g28970-d84029-reviews-the_georgetown_inn-washington_dc_district_of_columbia.html.json