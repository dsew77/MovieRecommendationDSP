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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

#Reading the movies metadata
#Change to True if running for the first time

md = pd.read_csv('movies_metadata.csv')
#Setting up genres into its text and id components
md['genres_text'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['genres_id'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['id'] for i in x] if isinstance(x, list) else [])

#%%



#Setting up a weighting function based on IMDB weighted rating
v = md['vote_count'].fillna(0).astype('int')
R = md['vote_average'].fillna(0).astype('float')

C = R.mean()  
#Find the minimum votes required based on the percentile
# A percentile of 95 means it takes the top 5% of movies by votes
percentile = 85
m = v.quantile(percentile/100)

md['wr'] = ((v/(v+m) * R) + (m/(v+m) * C) )

IMDB_list = md.sort_values('wr', ascending = False)

#Assign different weightings to different subjects
#From 1 - 10 (Not important to Very Important)
genre_weight = 10
genres_list = ['Action','Comedy']
#director
#actor
#budget
similar_movies_weight = 2
liked_movies = ['']

#Using Scikti TF-IDF to get similarites between movies
#Combine tagline and overview to create a description column
md['description'] = md['tagline'] + md['overview']
md['description'] = md['description'].fillna('')






titles = md['original_title']
#Set up the indices
indices = pd.Series(md.index, index=md['original_title'])

liked_movies = 'The Dark Knight'
index = indices[liked_movies][0]


#Initialsie the Recommended Rating
Recommend_Rating = [0 for i in range(len(md))]
for i in range(len(md)):
    for j in range(len(genres_list)):
        if genres_list[j] in md['genres_text'][i]:
            Recommend_Rating[i] = genre_weight/len(genres_list)

            
md['Recommended_Rating'] = Recommend_Rating + md['wr'] + similar_movies_weight*cosine_sim[index]
Recommended_list = md.sort_values('Recommended_Rating', ascending = False)



#%%
'''
Function create_cosine_similarity
Using TF-IDF check similarities between the descriptions of various movies 
Input:
Output:
    cosine_sim
'''
def create_cosine_similarity():
    #Initialise the TF-IDF Vectorizer
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    #Create the matrix of similarities
    tfidfMatrix = tf.fit_transform(md['description'])
    #Using Sklearn linear Kernel 
    cosine_sim = linear_kernel(tfidfMatrix, tfidfMatrix)

    return cosine_sim


cosine_sim = create_cosine_similarity()


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