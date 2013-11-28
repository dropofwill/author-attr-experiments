from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = ("The sky is blue",
			"The sun is bright",
			"The sun in the sky is bright",
			"We can see the shining sun, the bright sun")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print tfidf_matrix.shape

print cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

#x_test = vectorizer.transform(test_set)