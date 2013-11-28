from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import Bootstrap

from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

from scipy.stats import sem
from pprint import pprint
import numpy as np
import pylab as pl

# Calculates the mean of the scores with the standard deviation
def mean_sem(scores):
	return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)) 

# Load documents
docs = datasets.load_files(container_path="../../sklearn_data/problemH/")
X, y = docs.data, docs.target

# Select Features via Bag of Words approach without stop words
X = CountVectorizer(charset_error='ignore', stop_words='english', strip_accents='unicode', ).fit_transform(X)
#X = TfidfVectorizer(charset_error='ignore', stop_words='english', strip_accents='unicode', sublinear_tf=True, max_df=0.5).fit_transform(X)
n_samples, n_features = X.shape

'''

n_alphas = 10
n_iter = 10
cv = Bootstrap(n_samples, n_iter=n_iter)
train_scores = np.zeros((n_alphas, n_iter))
test_scores = np.zeros((n_alphas, n_iter))
alphas = np.logspace(-7,-1,n_alphas)
print alphas

for i, alpha in enumerate(alphas):
	for j, (train, test) in enumerate(cv):
		clf = MultinomialNB(alpha=alpha).fit(X[train], y[train])
		train_scores[i, j] = clf.score(X[train], y[train])
		test_scores[i, j] = clf.score(X[test], y[test])
'''

n_alphas = 10
n_iter = 5
cv = ShuffleSplit(n_samples, n_iter=n_iter, random_state=0)

train_scores = np.zeros((n_alphas, n_iter))
test_scores = np.zeros((n_alphas, n_iter))
alphas = np.logspace(-10, 0, n_alphas)

for i, alpha in enumerate(alphas):
    for j, (train, test) in enumerate(cv):
        clf = MultinomialNB(alpha=alpha).fit(X[train], y[train])
        train_scores[i, j] = clf.score(X[train], y[train])
        test_scores[i, j] = clf.score(X[test], y[test])


'''
n_gammas = 10
n_iter = 5
cv = ShuffleSplit(n_samples, n_iter=n_iter, train_size=500, test_size=500,
    random_state=0)

train_scores = np.zeros((n_gammas, n_iter))
test_scores = np.zeros((n_gammas, n_iter))
gammas = np.logspace(-7, -1, n_gammas)

for i, gamma in enumerate(gammas):
    for j, (train, test) in enumerate(cv):
        clf = SVC(C=10, gamma=gamma).fit(X[train], y[train])
        train_scores[i, j] = clf.score(X[train], y[train])
        test_scores[i, j] = clf.score(X[test], y[test])
'''