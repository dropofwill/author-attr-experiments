import nltk, random
from nltk.corpus import PlaintextCorpusReader, CategorizedPlaintextCorpusReader


# Build corpus for specific problem set
problem = 'problemA'
problem_root = nltk.data.find('corpora/AAAC/%s' % (problem))
problem_files = PlaintextCorpusReader(problem_root, '.*\.txt')


# Categorize corpus by author
auth_map = {}
for filename in problem_files.fileids():
	a_n =  filename[:3]
	auth_map[filename] =  [a_n]

# By the entire corpus
problem_cat = CategorizedPlaintextCorpusReader(problem_root, '.*\.txt', cat_map=auth_map)
documents = [(list(problem_cat.words(fileid)), category) 
				for category in problem_cat.categories() 
				for fileid in problem_cat.fileids(category)]
random.shuffle(documents)


# Word Frequency featureset
# Word freq accross corpus
all_words = nltk.FreqDist(words.lower() for words in problem_cat.words())
key_words = all_words.keys()[:2000]


# Compares whether a word from the keywords is in a document
def doc_features(doc):
	doc_words = set(doc)
	features = {}
	for word in key_words:
		features['contains(%s)' % word] = (word in doc_words)
	return features

featureset = [(doc_features(docs), categories) for (docs, categories) in documents]

test_len = len(featureset)/4
train_len = len(featureset) - len(featureset)/4

train_set, test_set = featureset[:train_len], featureset[test_len:]


classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(10)


'''
test_corp = PlaintextCorpusReader(problem_root, 'a01_t1.txt')
test_list = [(list(test_corp.words(fileid)), category) 
				for category in test_corp.categories() 
				for fileid in test_corp.fileids(category)]
print classifier.classify(doc_features(test_file))
'''