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
from sklearn.cross_validation import Bootstrap
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

###############################################################################

docs = datasets.load_files(container_path="../../sklearn_data/problemB")

# A/C
# use_idf=true
# alpha=0.001
# max_df= 1
# max_features = none

X, y = docs.data, docs.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



baseline = 1/float(len(list(np.unique(y))))

#bs = Bootstrap(n,test_size=0.3, random_state=42)

# define a pipeline combining a text feature extractor with a simple classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(charset_error='ignore')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(alpha=0.001))
])


# features to cross-check
parameters = {
    #'vect__max_df': (0.75, 1.0),
    #'vect__max_features': (None, 100, 5000, 10000),
    'vect__analyzer' : ('char', 'word'),
    #'vect__ngram_range': ((1, 1), (1, 2), (2,3), (1,3), (3,4), (1,5), (4,5)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (1, 0.5, 0.01, 0.001, 0.0001, 0.000001)
}

sub_dir = "Results/"
location = "results" + tm.strftime("%Y%m%d-%H%M%S") + ".txt"

scores = ['precision', 'recall']

with open( os.path.join(sub_dir, location), 'w+') as myFile:
    
    for score in scores:  
        # classifier
        grid_search = GridSearchCV(pipeline, parameters, cv=2, verbose=0, scoring=score)

        print("Performing grid search...")
        print("parameters:")
        pprint(parameters)
        t0 = time()


        # fit data to training set
        grid_search.fit(X_train, y_train)    


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

        myFile.write("Best score: %0.3f \n" % grid_search.best_score_)
        myFile.write("Baseline score: %0.3f \n" % baseline)
        myFile.write("Best parameters set: \n")
        best_parameters = grid_search.best_estimator_.get_params()

        for param_name in sorted(parameters.keys()):
            myFile.write("\t%s: %r \n" % (param_name, best_parameters[param_name]))

        myFile.write("All parameters tried: \n")
        all_parameters = grid_search.grid_scores_
        
        for params, mean_score, scores in sorted(grid_search.grid_scores_):
            myFile.write("\t \t %0.3f (+/-%0.03f) for %r \n" % (mean_score, scores.std() / 2, params))

    myFile.write("Detailed classification report: \n\n")
    y_true, y_pred = y_test, grid_search.predict(X_test)
    myFile.write(classification_report(y_true, y_pred))