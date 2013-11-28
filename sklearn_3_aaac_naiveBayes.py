from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

docs = datasets.load_files(container_path="../../sklearn_data/problemC/")

X, y = docs.data, docs.target

X = CountVectorizer(charset_error='ignore', stop_words='english').fit_transform(X)

n_samples, n_features = X.shape

#print MultinomialNB().fit(X, y).score(X,y)


X_train, X_test, y_train, y_test = train_test_split(X, y)

print X_train.shape
print y_train.shape

print X_test.shape
print y_test.shape

mnb = MultinomialNB().fit(X_train, y_train)
print mnb.score(X_test, y_test)