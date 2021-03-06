import ast
import pandas as pd
import numpy as np
from shutil import copyfile
import os


def parse_str_list(df):
    df['text_filtered_per_sentence'] = df['text_filtered_per_sentence'].apply(
        ast.literal_eval)
    df['aspect_filtered_per_sentence'] = df['aspect_filtered_per_sentence'].apply(
        ast.literal_eval)


def build_asp_rating(split):
    aspect_rating = split.replace(-1, np.nan).groupby(
        'hotel_url')[["service", "cleanliness", "value", "location", "rooms"]].mean()

    aspect_rating = aspect_rating.rename(columns={'service': 'aspect 0 ', 'cleanliness': 'aspect 1 ',
                                  'value': 'aspect 2 ', 'location': 'aspect 3 ', 'rooms': 'aspect 4 '}).reset_index()

    aspect_rating = pd.melt(aspect_rating, id_vars='hotel_url', value_vars=[
                            'aspect 0 ', 'aspect 1 ', 'aspect 2 ', 'aspect 3 ', 'aspect 4 ', ])

    aspect_rating['hotel_url_new'] = aspect_rating['variable'] + \
        aspect_rating['hotel_url']
    aspect_rating = aspect_rating[['hotel_url_new', 'value']].rename(
        columns={'hotel_url_new': 'hotel_url'})

    aspect_rating = aspect_rating.replace(np.nan, -1)
    aspect_rating.rename(columns={'value': 'rating'}, inplace=True)
    return aspect_rating


def groupByAspect(split):
    split_sentence = split[['hotel_url', 'text', 'text_filtered_per_sentence']].explode(
        'text_filtered_per_sentence')
    split_aspect = split[['hotel_url', 'text', 'aspect_filtered_per_sentence']].explode(
        'aspect_filtered_per_sentence')
    split_sentence.reset_index(drop=True, inplace=True)
    split_aspect.reset_index(drop=True, inplace=True)
    split_exploded = split_aspect.join(
        split_sentence[['text_filtered_per_sentence']])

    grouped = pd.DataFrame(split_exploded.groupby(['hotel_url', 'aspect_filtered_per_sentence'])[
                           'text_filtered_per_sentence'].apply(list))
    grouped.reset_index(inplace=True)
    grouped['text_filtered_per_sentence'] = grouped['text_filtered_per_sentence'].apply(
        lambda data: [data[x:x+5] for x in range(0, len(data), 5)])
    grouped.head()

    grouped = grouped.explode('text_filtered_per_sentence').reset_index()
    grouped['text'] = grouped['text_filtered_per_sentence'].apply(
        lambda x: ''.join(x))
    grouped = grouped.rename(columns={'hotel_url': 'hotel_url_original'})
    grouped['hotel_url'] = grouped['hotel_url_original']
    grouped = grouped[['index', 'hotel_url', 'aspect_filtered_per_sentence',
           'text_filtered_per_sentence', 'text', 'hotel_url_original']]

    grouped['hotel_url'] = 'aspect ' + \
        grouped['aspect_filtered_per_sentence'].apply(
            str)+' '+grouped['hotel_url_original']
    return grouped


if __name__ == '__main__':
    test = pd.read_csv('./test.csv')
    train = pd.read_csv('./train.csv')
    dev = pd.read_csv('./dev.csv')

    parse_str_list(test)
    parse_str_list(train)
    parse_str_list(dev)

    test_asp_rating = build_asp_rating(test)
    train_asp_rating = build_asp_rating(train)
    dev_asp_rating = build_asp_rating(dev)

    test = groupByAspect(test)
    train = groupByAspect(train)
    dev = groupByAspect(dev)

    # merge rating dataframe with RSAR review dataframe
    test = pd.merge(test, test_asp_rating, left_on='hotel_url',
                    right_on='hotel_url', how='left')
    train = pd.merge(train, train_asp_rating, left_on='hotel_url',
                     right_on='hotel_url', how='left')
    dev = pd.merge(dev, dev_asp_rating, left_on='hotel_url',
                   right_on='hotel_url', how='left')

    
    # round float to 1-5 integers
    test=test[test['rating']!=-1]
    train=train[train['rating']!=-1]
    dev=dev[dev['rating']!=-1]

    test['rating']=test['rating'].apply(np.round)
    train['rating']=train['rating'].apply(np.round)
    dev['rating']=dev['rating'].apply(np.round)




    "executed in 51.6s"
    tot=pd.concat([dev,test,train])

    "executed in 1m 1.82s"
    df=tot.sort_values('hotel_url')[['hotel_url','text','rating']]

    dataset_path='../hotel_mask_sing_asp_dataset/'
    df.to_json(os.path.join(dataset_path,'review.json'),orient='records')

    src=os.path.join(dataset_path,'review.json')
    dst=os.path.join(dataset_path,'business.json')
    copyfile(src, dst)





