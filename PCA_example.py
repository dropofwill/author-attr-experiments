from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import RandomizedPCA
from itertools import cycle
import numpy as np
from sklearn.cross_validation import ShuffleSplit
import string
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

docs = datasets.load_files(container_path="../../sklearn_data/problemF")
X, y = docs.data, docs.target

X = TfidfVectorizer(charset_error='ignore', stop_words='english', analyzer='char', ngram_range=(2,4), strip_accents='unicode', sublinear_tf=True, max_df=0.5).fit_transform(X)
n_samples, n_features = X.shape

#print y

pca = RandomizedPCA(n_components=2)
X_pca = pca.fit_transform(X)

#print X_pca.shape

colors = ['b', 'w', 'r']
markers = ['+', 'o', '^']
# for i, c, m in zip(np.unique(y), cycle(colors), cycle(markers)):
#     pl.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=c, marker=m, label=i, alpha=0.5)

# _ = pl.legend(loc='best')

# pl.show()

fig = pl.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = RandomizedPCA(n_components=3).fit_transform(X)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, s=100)

ax.set_title("A PCA Reduction of High Dimensional Data to 3 Dimensions")
ax.set_xlabel("1st")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd")
ax.w_zaxis.set_ticklabels([])
pl.legend(loc='best')

pl.show()


# n_alphas = 10
# n_iter = 5
# cv = ShuffleSplit(n_samples, n_iter=n_iter,random_state=0)

# train_scores = np.zeros((n_alphas, n_iter))
# test_scores = np.zeros((n_alphas, n_iter))
# gammas = np.logspace(-7, -1, n_gammas)

# for i, gamma in enumerate(gammas):
#     for j, (train, test) in enumerate(cv):
#         clf = SVC(C=10, gamma=gamma).fit(X[train], y[train])
#         train_scores[i, j] = clf.score(X[train], y[train])
#         test_scores[i, j] = clf.score(X[test], y[test])