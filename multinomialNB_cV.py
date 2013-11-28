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
#X = CountVectorizer(charset_error='ignore', stop_words='english', strip_accents='unicode', ).fit_transform(X)
X = TfidfVectorizer(charset_error='ignore', stop_words='english', strip_accents='unicode', sublinear_tf=True, max_df=0.5).fit_transform(X)
n_samples, n_features = X.shape

'''
# sklearn's grid search
parameters = { 
	'alpha': np.logspace(-25,0,25)
}
	#pprint(parameters)

bv = Bootstrap(n_samples, n_iter=10, test_size=0.3, random_state=42)
mnb_gv = GridSearchCV(MultinomialNB(), parameters, cv=bv,)
mnb_gv.fit(X, y)
print mnb_gv.best_params_
print mnb_gv.best_score_

mnb_best_score = mnb_gv.best_score_
'''

# CV with Bootstrap
'''
mnb = MultinomialNB(alpha=mnb_best_score)
#bv = Bootstrap(n_samples, n_iter=100, test_size=0.2, random_state=42)
boot_scores = cross_val_score(mnb, X, y, cv=bv)
print mean_sem(boot_scores)
'''


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