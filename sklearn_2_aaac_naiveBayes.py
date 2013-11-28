from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import RandomizedPCA

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse

from itertools import cycle

import matplotlib
matplotlib.use('TkAgg')
import pylab as pl
import numpy as np

docs = datasets.load_files(container_path="../../sklearn_data/problemC/")

X, y = docs.data, docs.target

X = CountVectorizer(charset_error='ignore', stop_words='english').fit_transform(X)

n_samples, n_features = X.shape

print X.shape
print n_samples
print n_features

X_pca = RandomizedPCA(n_components=2).fit_transform(X)
print X_pca.shape

pl.show()
pl.draw()

colors = ['b','g','r','c','m','y','k']
for i, c in zip(np.unique(y), cycle(colors)):
	pl.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label=i, alpha=0.5)

pl.legend(loc="best")