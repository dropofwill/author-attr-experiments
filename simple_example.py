from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
from sklearn.naive_bayes import MultinomialNB

#train_set = ("The sky is blue.", "The sun is bright.", "The stars are bright")
#test_set = ("The mountains are covered with trees.", "Despite the cold the trees are green.")

docs = datasets.load_files(container_path="SimpleData/")
X, y = docs.data, docs.target

# print "Training Set:" 
# for i in train_set:
# 	print "\t" + i
# print "Testing Set:"
# for e in test_set:
# 	print "\t" + e

count_vectorizer = CountVectorizer(min_df=1)
matrix = count_vectorizer.fit_transform(X)

n_samples, n_features = matrix.shape


print matrix.shape

print count_vectorizer.vocabulary_
print count_vectorizer.fit_transform(X).todense()

#mnb = MultinomialNB().fit(matrix, y)
#print mnb.score(X, y)

# tf_idf = TfidfTransformer()
# tf_idf.fit(freq_matrix)

# tf_idf_matrix = tf_idf.transform(freq_matrix)
# print tf_idf_matrix.todense()