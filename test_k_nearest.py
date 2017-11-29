import pandas as pd

import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB



category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]

docs_new = input()
docs_new = [docs_new]

#LOAD MODEL
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("model2.pkl","rb"))

X_new_counts = loaded_vec.transform(docs_new)
X_new_tfidf = loaded_tfidf.transform(X_new_counts)
predicted = loaded_model.predict(X_new_tfidf)


print(category_list[predicted[0]])