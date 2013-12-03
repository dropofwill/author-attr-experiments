from __future__ import print_function
from pprint import pprint
from time import time
import logging
import numpy as np
import os
import time as tm

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

###############################################################################

docs = datasets.load_files(container_path="../../sklearn_data/problemC")
X, y = docs.data, docs.target

baseline = 1/float(len(list(np.unique(y))))

# define a pipeline combining a text feature extractor with a simple classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(charset_error='ignore')),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', MultinomialNB())
])

# features to cross-check
parameters = {
    'vect__max_df': (0.75, 1.0),
    'vect__max_features': (None, 5000, 10000),
    #'vect__analyzer' : ('char', 'word'),
    'vect__ngram_range': ((1, 1), (1, 2), (2,3)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (1.0, 0.0001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (50, 100),
}

# classifier
grid_search = GridSearchCV(pipeline, parameters, verbose=1)

print("Performing grid search...")
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(X, y)
print("done in %0.3fs" % (time() - t0))
print()

improvement = (grid_search.best_score_ - baseline) / baseline

print("Best score: %0.3f" % grid_search.best_score_)
print("Baseline score: %0.3f" % baseline)
print("Improved: %0.3f over baseline" % improvement)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

sub_dir = "Results/"
location = "results" + tm.strftime("%Y%m%d-%H%M%S") + ".txt"

with open( os.path.join(sub_dir, location), 'w+') as myFile:
    myFile.write("Best score: %0.3f \n" % grid_search.best_score_)
    myFile.write("Baseline score: %0.3f \n" % baseline)
    myFile.write("Best parameters set: \n")
    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):
        myFile.write("\t%s: %r \n" % (param_name, best_parameters[param_name]))
