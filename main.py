# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 09:22:29 2021

@author: Daniel Wong
Movie Recommendation data science project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

#Reading the movies metadata
#Change to True if running for the first time

md = pd.read_csv('movies_metadata.csv')


#%%
#Setting up genres into its text and id components
md['genres_text'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['genres_id'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['id'] for i in x] if isinstance(x, list) else [])


#Setting up a weighting function based on IMDB weighted rating
vote_counts = md['vote_count'].fillna(0).astype('int')












#%%
#Used to get the unique id and genres

#Assigning each id to its text respectively (could probably use the original)
unique = []
unique_genre = []

for i in range(len(md)):
    for j in range(len(md['genres_id'][i])):
        
        if md['genres_id'][i][j] not in unique:
            unique.append(md['genres_id'][i][j])
            unique_genre.append(md['genres_text'][i][j])
            
            
genres = [unique, unique_genre]