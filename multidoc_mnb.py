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
import string
import matplotlib.pyplot as plt

# Calculates the mean of the scores with the standard deviation
def mean_sem(scores):
	return ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)) 


def test_docs(dir):
	# Load documents
	docs = datasets.load_files(container_path="../../sklearn_data/"+dir)
	X, y = docs.data, docs.target

	baseline = 1/float(len(list(np.unique(y))))

	# Select Features via Bag of Words approach without stop words
	#X = CountVectorizer(charset_error='ignore', stop_words='english', strip_accents='unicode', ).fit_transform(X)
	X = TfidfVectorizer(charset_error='ignore', stop_words='english', analyzer='char', ngram_range=(2,4), strip_accents='unicode', sublinear_tf=True, max_df=0.5).fit_transform(X)
	n_samples, n_features = X.shape


	# sklearn's grid search
	parameters = { 'alpha': np.logspace(-100,0,10)}

	bv = Bootstrap(n_samples, n_iter=10, test_size=0.3, random_state=42)
	mnb_gv = GridSearchCV(MultinomialNB(), parameters, cv=bv,)
	#scores = cross_val_score(mnb_gv, X, y, cv=bv)
	mnb_gv.fit(X, y)
	mnb_gv_best_params = mnb_gv.best_params_.values()[0]
	print mnb_gv.best_score_
	print mnb_gv_best_params

	# CV with Bootstrap
	mnb = MultinomialNB(alpha=mnb_gv_best_params)
	boot_scores = cross_val_score(mnb, X, y, cv=bv)
	print mean_sem(boot_scores)

	improvement = (mnb_gv.best_score_ - baseline) / baseline

	rand_baseline.append(baseline)
	test_results.append([mnb_gv.best_score_])
	com_results.append(improvement)
	sem_results.append(sem(boot_scores))


def graph(base_list, results_list, com_list, arange):
	N=arange
	base=np.array(base_list)
	res=np.array(results_list)
	com = np.array(com_list)
	ind = np.arange(N)    # the x locations for the groups
	width = 0.3       # the width of the bars: can also be len(x) sequence

	#fig, ax = plt.sublots()

	p1 = plt.bar(ind, base, width, color='r')
	p2 = plt.bar(ind+0.3, res, width, color='y')
	p3 = plt.bar(ind+0.6, com, width, color='b')

	plt.rcParams['figure.figsize'] = 10, 7.5
	plt.rcParams['axes.grid'] = True
	plt.gray()

	plt.ylabel('Accuracy')
	plt.title('AAAC Problem Accuracy')
	plt.yticks(np.arange(0,3,30))
	plt.xticks(np.arange(0,13,13))
	#plt.set_xticks(('A','B','C','D','E','F','G','H','I','J','K','L','M'))
	plt.legend( (p1[0], p2[0], p3[0]), ('Baseline', 'Algorithm', 'Improvement'))

	plt.show()


rand_baseline = list()
test_results = list()
sem_results = list()
com_results = list()

#test_docs("problemA")

for i in string.uppercase[:13]:
	test_docs("problem"+i)

#graph(rand_baseline,test_results,com_results,13)




import os
import time as tm

sub_dir = "Results/"
location = "multiDoc" + tm.strftime("%Y%m%d-%H%M%S") + ".txt"

with open(os.path.join(sub_dir, location), 'w') as myFile:
	myFile.write(str(rand_baseline))
	myFile.write("\n")
	myFile.write(str(test_results))
	myFile.write("\n")
	myFile.write(str(sem_results))
	myFile.write("\n")
	myFile.write(str(com_results))






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