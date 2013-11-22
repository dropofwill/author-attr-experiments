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

#
def doc_features(doc):
	doc_words = set(doc)
	features = {}
	for word in key_words:
		features['contains(%s)' % word] = (word in doc_words)
	return features

#print problem_cat.categories()
#for category in problem_cat.categories():
#	print problem_cat.fileids(category)