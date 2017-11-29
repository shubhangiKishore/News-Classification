
import pandas as pd

import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import svm 
from sklearn.neural_network import MLPClassifier




category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

X_train_tfidf = pickle.load(open("tfidf.pkl","rb"))
training_data = pd.read_csv("train_data", sep = ',')
print(training_data.data.shape)



count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))

'''
clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)

#SAVE MODEL
pickle.dump(clf, open("model_nv.pkl", "wb"))
'''

'''

clf_svm = svm.LinearSVC()

clf_svm.fit(X_train_tfidf, training_data.flag)

'''

#clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)


clf_neural.fit(X_train_tfidf, training_data.flag)

# load pickled files 
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("model.pkl","rb"))


# TAKE INPUT FROM USER
docs_new = input()
docs_new = [docs_new]
X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)

# For testing.
#predicted = clf_svm.predict(X_new_tfidf)

predicted = clf_neural.predict(X_new_tfidf)



print(category_list[predicted[0]])

#pickle.dump(clf_svm, open("model_svm.pkl", "wb"))

pickle.dump(clf_neural, open("model_neural.pkl", "wb"))

'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = input()
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("model.pkl","rb"))

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print(category_list[predicted[0]]


from sklearn.neighbors import KNeighborsClassifier

clf_k = KNeighborsClassifier(n_neighbors=32)

model2 = clf_k.fit(X_train_tfidf, training_data.flag)
#save model
pickle.dump(model2, open("model2.pkl", "wb"))

loaded_model = pickle.load(open("model.pkl","rb"))



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:27:05 2017

@author: vijaynandwani
"""

"""
To create sub directories of news with one news file

with open('news', 'r') as f:
    text = f.read()
    news = text.split("\n\n")
    count = {'sport': 0, 'world': 0, "us": 0, "business": 0, "health": 0, "entertainment": 0, "sci_tech": 0}
    for news_item in news:
        lines = news_item.split("\n")
        print(lines[6])
        file_to_write = open('data/' + lines[6] + '/' + str(count[lines[6]]) + '.txt', 'w+')
        count[lines[6]] = count[lines[6]] + 1
        file_to_write.write(news_item)  # python will convert \n to os.linesep
        file_to_write.close()
"""

import pandas
import glob
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
directory_list = ["data/sport/*.txt", "data/world/*.txt","data/us/*.txt","data/business/*.txt","data/health/*.txt","data/entertainment/*.txt","data/sci_tech/*.txt",]

text_files = list(map(lambda x: glob.glob(x), directory_list))
text_files = [item for sublist in text_files for item in sublist]

training_data = []


for t in text_files:
    f = open(t, 'r')
    f = f.read()
    t = f.split('\n')
    training_data.append({'data' : t[0] + ' ' + t[1], 'flag' : category_list.index(t[6])})
    
training_data[0]

training_data = pandas.DataFrame(training_data, columns=['data', 'flag'])

print(training_data.data.shape)


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))

#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))


clf = MultinomialNB().fit(X_train_tfidf, training_data.flag)

#SAVE MODEL
pickle.dump(clf, open("model.pkl", "wb"))



# For testing.

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = input()
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("model.pkl","rb"))

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)

print(category_list[predicted[0]]


from sklearn.neighbors import KNeighborsClassifier

clf_k = KNeighborsClassifier(n_neighbors=32)

model2 = clf_k.fit(X_train_tfidf, training_data.flag)
#save model
pickle.dump(model2, open("model2.pkl", "wb"))

loaded_model = pickle.load(open("model.pkl","rb"))

'''

