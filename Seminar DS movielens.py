#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# In[2]:


pip install scikit-learn


# In[3]:


import pandas as pd
import math
from sklearn.decomposition import NMF


# data visualisation and manipulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


# ## Loading the data sets

# In[6]:


train_movies=pd.read_csv("movies.csv")


# In[5]:


train_ratings=pd.read_csv("ratings.csv")


# ## We see the overview of all feilds in the rating table

# In[7]:


df=train_ratings.copy()
df.head()


# ## 1.Exploring the dataset

# ####  Note that this file contains the ratings given by our set of users to different movies. In all it contains total 100K ratings; to be exact 1000004.
# 
# #### Note that for 671 users and 9066 movies we can have a maximum of 671*9066 = 6083286 ratings. 
# #### But note that we have only 100004 ratings with us. 
# #### Hence the utility matrix has only about 1.6 % of the total values. 
# #### Thus it can be concluded that it is quite sparse.

# In[11]:


df['rating'].min() # minimum rating


# In[10]:


df['rating'].max() # maximum rating


# In[16]:


#Encoding the columns
df.userId = df.userId.astype('category').cat.codes.values
df.movieId = df.movieId.astype('category').cat.codes.values


# ## 2.Creating the Utility Matrix

# In[17]:


index=list(df['userId'].unique())
columns=list(df['movieId'].unique())
index=sorted(index)
columns=sorted(columns)
 
util_df=pd.pivot_table(data=df,values='rating',index='userId',columns='movieId')
# Nan implies that user has not rated the corressponding movie.


# In[19]:


util_df.head()
#util_df.fillna(0)


# 1) This is the utility matrix; for each of the 671 users arranged rowwise; each column shows the rating of the movie given by a particular user.
# 
# 2) Note that majority of the matrix is filled with 'Nan' which shows that majority of the movies are unrated by many users.
# 
# 3) For each movie-user pair if the entry is NOT 'Nan' the vaue indicates the rating given by user to that corressponding movie. 
# 
# 4) For now I am gonna fill the 'Nan' value with value '0'. But note that this just is just indicative, a 0 implies NO RATING and doesn't mean that user has rated 0 to that movie. It doesn't at all represent any rating.
# 

# # from sklearn.decomposition import NMF
# #perform a join between them ? 
# 
# 
# # Example usage
# #model = NMF(n_components=2)
# #W = model.fit_transform(X)  # Where X is your data matrix after merging 2 tables
# #H = model.components_
# 

# In[ ]:



# File paths
#ratings_file = "/Users/orlikagan/Desktop/MSC_DS/סמסטר ב/סמינר DS/data/ratings.csv"
#movies_file = "/Users/orlikagan/Desktop/MSC_DS/סמסטר ב/סמינר DS/data/movies.csv"

# Load the datasets
#ratings = pd.read_csv(ratings_file)
#movies = pd.read_csv(movies_file)

# Merge datasets to get a user-item matrix
data = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating', fill_value=0)

# Apply NMF
model = NMF(n_components=2, random_state=42)
W = model.fit_transform(user_movie_matrix)
H = model.components_


# In[10]:


# Merge datasets to get a user-item matrix
data = pd.merge(ratings, movies, on='movieId')
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating', fill_value=0)


# In[21]:


# Print the results
print("W (User feature matrix):")
print(W)
print("\nH (Movie feature matrix):")
print(H)


# In[20]:


#print(user_movie_matrix).head(100)


# In[ ]:





# In[ ]:





# In[ ]:




