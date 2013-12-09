import numpy as np
import os
import time as tm

from sklearn import datasets
from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import SparsePCA

# A / English / Fixed-Topic Essays / 13 labels / min 3, max 4
# Problem A : alpha 0.0001; max_features None, word (4,5) 
#             NA: use_idf, max_df

# B / English / Free-Topic Essays / 13 labels / 3, 4
# Problem B : alpha 0.00001; use_idf true, max_df 0.75,
#             max_features None, word (1,1)

# C / English / 100,000char of 19th American Novels / 5 / 4, 6
# Problem C : alpha 0.00001; use_idf true, max_df 0.75/0.5,
#             max_features None, word (3,4), (3,5)

# D / English / 1st Act of Shakespeare-era Plays / 3 / 4, 6
# Problem D : alpha 0.00001; use_idf true, max_df 0.75,
# 0.927      max_features None, char, permutations of ngrams 1-3

# E / English / Entire Shakespeare-era Plays / 3 / 4, 6 
# Problem E : alpha 0.00001; use_idf true, max_df 0.75,
# 0.842       max_features None, char, permutations of ngrams 1-3

# F / Middle-English / Paston Letters / 3 / 23, 23
# Problem F : alpha 0.00001; use_idf true, max_df 0.75
# 0.963        max_features None, word, (4,5), (1,5), (1,4)

        # G / English / Edgar Burrows Novels pre1914-post1920 / 2 / 5,5
        # Problem G : alpha 0.00001; use_idf true, max_df 0.75
        # 0.5-0.6     max_features None, any

# H / English / Transcript of commitee meetings / 3 / 5,6
# Problem H : alpha 0.01; use_idf true, max_df 1
# 0.8         max_features None, word/char (1,1) (3,5), (3,4)

        # I / French / Novels by Dumas & Hugos / 2 / 4,5
        # Problem I : alpha 0.0001; use_idf true, max_df 0.75
        #    max_features None, char (1,1) (1,2), (1,3)

        # J / French / Plays by Dumas & Hugos / 2 / 3,4
        # Problem J : alpha 0.01; use_idf true, max_df 1
        #    max_features None, word/char (1,1) (3,5), (3,4)

# K / Serbian-Slavonic / Exercpts from the Lives of Kings and Archbishops / 3 / 3,4
# Problem K : alpha 0.0001; use_idf true, max_df 0.75
# 0.75   max_features None, word (4,5)

# L / Latin / Elegaic Poems from Classical Latin / 4 / 6,6
# Problem L : alpha 0.01; use_idf true, max_df 0.75
# 0.965   max_features None, char (4,5)

# M / Dutch / Fix-Topic Dutch Essays / 8 / 9,9
# Problem M : alpha 0.01; use_idf true, max_df 0.75
# 0.965   max_features None, char (4,5)

docs = datasets.load_files(container_path="../../sklearn_data/problemM")

X, y = docs.data, docs.target

baseline = 1/float(len(list(np.unique(y))))

# Split the dataset into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define a pipeline combining a text feature extractor/transformer with a classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(decode_error='ignore')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB(alpha=0.01))
])

# features to cross-check
parameters = {
    #'vect__max_df': (0.5, 0.75, 1),
    #'vect__max_features': (None, 100, 5000),
    'vect__analyzer' : ('char', 'word'),
    'vect__ngram_range': ((1, 1), (1, 2), (2,2), (2,3), (1,3), (1,4), (3,4), (1,5), (4,5), (3,5)),
    #'vect__ngram_range': ((1, 1), (1, 2), (1,3)),  # unigrams or bigrams or ngrams
    #'vect__ngram_range': ((3,4), (3, 5), (4,5)),
    #'tfidf__use_idf': (True, False),
    #'clf__alpha': (1, 0.5, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001),
    #'clf__alpha': (0.01, 0.001, 0.0001)
}

scores = ['precision', 'recall']

sub_dir = "Results/"
location = "results" + tm.strftime("%Y%m%d-%H%M%S") + ".txt"

with open( os.path.join(sub_dir, location), 'w+') as f:
    for score in scores:
        f.write("%s \n" % score)
        clf = GridSearchCV(pipeline, parameters, cv=5, scoring=score, verbose=0)
        clf.fit(X_train, y_train)
        improvement = (clf.best_score_ - baseline) / baseline

        f.write("Best parameters from a %s stand point:\n" % score)
        f.write("Best score: %0.3f \n" % clf.best_score_)
        f.write("Baseline score: %0.3f \n" % baseline)
        f.write("Improved: %0.3f over baseline \n" % improvement)

        f.write("\n\nGrid scores from a %s stand point:\n" % score)
        
        for params, mean_score, scores in clf.grid_scores_:
            f.write("%0.3f (+/-%0.03f) for %r \n" % (mean_score, scores.std() / 2, params))
        f.write("\n\n")

    f.write("\n\nDetailed classification report:\n")
    f.write("The model is trained on the full development set.\n")
    f.write("The scores are computed on the full evaluation set.\n")
    
    y_true, y_pred = y_test, clf.best_estimator_.predict(X_test)
    
    f.write(classification_report(y_true, y_pred))