# MOVIE RECOMMENDER

In this notebook, I have built a movie recommendation hybrid system using
THE MOVIE DATASET from kaggle

Built using content based recommender and collaborative filtering.

## THE FOLLOWING LIBRARIES WERE USED:

from surprise import SVD  
from surprise import Dataset  
from surprise.model_selection import cross_validate  
from surprise import Reader, Dataset  
import numpy as np # linear algebra  
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)  
from ast import literal_eval  
from nltk.stem.snowball import SnowballStemmer  
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity  

## STRATING WITH:

### SIMPLE RECOMMENDER:
This system used the TMDB Vote Count, Vote Averages and generes to build Top Movies Charts applying the weightage rating formula.



### CONTENT BASED RECOMMENDER: 
Built two content based engines, one that took movie overview and taglines as input and the other which took metadata such as cast, crew, genre and keywords to come up with predictions. 

USED THE TF-IDF(term frequency-inverse document frequency), a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.

The I used the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. 

FInally changed its behavior by using linear kernel which is often recommended for text classification.
Used sklearn's linear_kernel() instead of cosine_similarities() since it is faster.

Used Count Vectorizer is a way to convert a given set of strings into a frequency representation.
Basically only counting the number of times a word appears in the document which results in biasing in favour of most frequent words.
TF-IDF is a better option as count vectorizer ends up in ignoring rare words which could have helped is in processing our data more efficiently.



### COLLABORATIVE FILTERING:
The above engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who she/he is.

Therefore we use a technique called Collaborative Filtering to make recommendations to Movie Watchers.
We use the RMSE ROOT MEAN SQUARE ERROR to solve some of the porblems our model faced and hence to increase the accuracy.
LESSER THE RMSE, BETTER THE ACCURACY.
Her I used the SUPRISE LIBRARY to implement SVD.



### HYBRID SYSTEM:
This section brings together techniques we have implemented in the content based and collaborative filter based engines to finally recommend the top 10 movies suggestions to a perticular user based on his interest.


## NOTE:
A task given be IEEE ROUND 2 recruitments.
I made this project as a beginner in Machine Learning so I am unable to predict the accuracy of the model and have many doubts regarding some of the code cells in the project. I worked to the best of my ability to understand the various algosrithms and libraries used in the project.
