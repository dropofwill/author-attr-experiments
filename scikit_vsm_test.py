from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_set = ("The sky is blue.", "The sun is bright.")
test_set = ("The sun in the sky is bright.", "We can see the shining sun, the bright sun.")

count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(train_set)

freq_matrix = count_vectorizer.transform(test_set)
freq_matrix.todense()

tf_idf = TfidfTransformer()
tf_idf.fit(freq_matrix)

tf_idf_matrix = tf_idf.transform(freq_matrix)
tf_idf_matrix.todense()