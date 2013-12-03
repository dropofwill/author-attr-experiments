from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_set = ("The sky is blue.", "The sun is bright.", "The stars are bright")
test_set = ("The sun in the sky is bright.", "We can see the shining sun, the bright sun.")

print "Training Set:" 
for i in train_set:
	print "\t" + i
print "Testing Set:"
for e in test_set:
	print "\t" + e

count_vectorizer = CountVectorizer(min_df=1)
print count_vectorizer
count_vectorizer.fit_transform(train_set)

print count_vectorizer.vocabulary_
print count_vectorizer.fit_transform(train_set).todense()

freq_matrix = count_vectorizer.transform(test_set)

print freq_matrix.todense()

tf_idf = TfidfTransformer()
tf_idf.fit(freq_matrix)

tf_idf_matrix = tf_idf.transform(freq_matrix)
print tf_idf_matrix.todense()