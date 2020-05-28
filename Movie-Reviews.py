# -*- coding: utf-8 -*-
"""
Created on Fri May 29 01:46:48 2020

@author: Brij
"""
## MOVIE-REVIEWS

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg
from nltk.corpus import wordnet
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import os 
import nltk.corpus

print(os.listdir(nltk.data.find('corpora')))

from nltk.corpus import movie_reviews
print(movie_reviews.categories())

print(len(movie_reviews.fileids('pos')))
print()
print(movie_reviews.fileids('pos'))
neg_rev= movie_reviews.fileids('neg')
print(len(neg_rev))
print(neg_rev)
rev= nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
print(rev)

rev_list=[]

for rev in neg_rev:
    rev_text_neg= rev = nltk.corpus.movie_reviews.words(rev)
    review_one_string= " ".join(rev_text_neg)
    review_one_string=review_one_string.replace(' ,',',')
    review_one_string=review_one_string.replace(' .','.')
    review_one_string=review_one_string.replace("\' ","''")
    rev_list.append(review_one_string)
    
print(len(rev_list))
pos_rev=movie_reviews.fileids('pos')
for rev in pos_rev:
    rev_text_pos= rev = nltk.corpus.movie_reviews.words(rev)
    review_one_string= " ".join(rev_text_neg)
    review_one_string=review_one_string.replace(' ,',',')
    review_one_string=review_one_string.replace(' .','.')
    review_one_string=review_one_string.replace("\' ","''")
    rev_list.append(review_one_string)
print(len(rev_list))
print(rev_list)

neg_target=np.zeros((1000,),dtype=np.int)
pos_target=np.ones((1000,),dtype=np.int)

target_list=[]
for neg_tar in neg_target:
    target_list.append(neg_tar)
for pos_tar in pos_target:
    target_list.append(pos_tar)

print(len(target_list))   
y= pd.Series(target_list) 
type(y)
print(y.head())

# for creating features use countVectorizer or bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vect= CountVectorizer(lowercase=True,stop_words='english',min_df=2)
X_count_vect=count_vect.fit_transform(rev_list)
print(X_count_vect.shape)
X_names=count_vect.get_feature_names()
print(X_names)
X_count_vect=pd.DataFrame(X_count_vect.toarray(),columns=X_names)
print(X_count_vect.shape)
print(X_count_vect.head())

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X_train_cv,X_test_cv,y_train_cv,y_test_cv=train_test_split(X_count_vect,y,test_size=.25,random_state=5)
print(X_train_cv.shape)
print(X_test_cv.shape)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
y_pred_gnb=gnb.fit(X_train_cv,y_train_cv).predict(X_test_cv)
print(accuracy_score(y_test_cv,y_pred_gnb))

from sklearn.naive_bayes import MultinomialNB
clf_cv=MultinomialNB()
clf_cv.fit(X_train_cv,y_train_cv)
y_pred=clf_cv.predict(X_test_cv)
print(type(y_pred))
print(accuracy_score(y_test_cv,y_pred))