
import pandas as pd

import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import svm 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split





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


clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)

clf_neural.fit(X_train, y_train)

#clf_neural.fit(X_train_tfidf, training_data.flag)

# load pickled files 
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl","rb"))
loaded_model = pickle.load(open("model.pkl","rb"))




predicted = clf_neural.predict(X_test)

result_softmax = pd.DataFrame( {'true_labels': y_test,'predicted_labels': predicted})

result_softmax.to_csv('res_softmax.csv', sep = ',')





print(category_list[predicted[0]])


pickle.dump(clf_neural, open("model_neural.pkl", "wb"))
