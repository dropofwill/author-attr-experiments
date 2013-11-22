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

# By entire corpus
'''
problem_cat = CategorizedPlaintextCorpusReader(problem_root, '.*\.txt', cat_map=auth_map)

# Sort documents randomly in an array
documents = [(list(problem_cat.words(fileid)), category) 
				for category in problem_cat.categories() 
				for fileid in problem_cat.fileids(category)]
random.shuffle(documents)


# Word Frequency featureset
# Word freq accross corpus
all_words = nltk.FreqDist(words.lower() for words in problem_cat.words())
key_words = all_words.keys()[:3000]

# Compares whether a word from the keywords is in a document
def doc_features(doc):
	doc_words = set(doc)
	features = {}
	for word in key_words:
		features['contains(%s)' % word] = (word in doc_words)
	return features
'''

# By test vs. train

problem_test = CategorizedPlaintextCorpusReader(problem_root, '.*\d\.txt', cat_map=auth_map)
problem_train = CategorizedPlaintextCorpusReader(problem_root, '.*\D\.txt', cat_map=auth_map)

# Sort documents randomly in an array
documents_test = [(list(problem_test.words(fileid)), category) 
				for category in problem_test.categories() 
				for fileid in problem_test.fileids(category)]
documents_train = [(list(problem_train.words(fileid)), category) 
				for category in problem_train.categories() 
				for fileid in problem_train.fileids(category)]
random.shuffle(documents_test)
random.shuffle(documents_train)