from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import Bootstrap
from sklearn.naive_bayes import MultinomialNB

from scipy.stats import sem

import numpy as np

docs = datasets.load_files(container_path="../../sklearn_data/problemC/")

X, y = docs.data, docs.target

X = CountVectorizer(charset_error='ignore', stop_words='english').fit_transform(X)

n_samples, n_features = X.shape


# CV with Bootstrap
mnb = MultinomialNB()
bv = Bootstrap(n_samples, n_iter=100, test_size=0.2, random_state=0)
boot_scores = cross_val_score(mnb, X, y, cv=bv)
print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(boot_scores), sem(boot_scores)) 


# CV with ShuffleSpit
'''
cv = ShuffleSplit(n_samples, n_iter=100, test_size=0.2, random_state=0)
test_scores = cross_val_score(mnb, X, y, cv=cv)
print np.mean(test_scores)
'''

# Single run through
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print X_train.shape
print y_train.shape

print X_test.shape
print y_test.shape

mnb = MultinomialNB().fit(X_train, y_train)
print mnb.score(X_test, y_test)
'''