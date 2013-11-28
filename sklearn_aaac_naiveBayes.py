from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse

data = datasets.load_files(container_path="../../sklearn_data/problemC/")

count_vectorizer = CountVectorizer(charset_error='ignore')
docs_train = [open(f).read() for f in data.filenames]
X_train_counts = count_vectorizer.fit_transform(docs_train)

X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)

mnb = MultinomialNB().fit(X_train_tfidf, data.target)

'''
tfidf_vectorizer = TfidfVectorizer(charset_error='ignore', stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data.data)

X, y = train_test_split(tfidf_matrix)

print tfidf_matrix

#gnb = MultinomialNB().fit(X, data.target)
'''