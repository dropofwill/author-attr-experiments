import nltk, random
from nltk.corpus import PlaintextCorpusReader, CategorizedPlaintextCorpusReader

problem = 'problemA'
problem_root = nltk.data.find('corpora/AAAC/%s' % (problem))
problem_files = PlaintextCorpusReader(problem_root, '.*\.txt')

auth_map = {}

for filename in problem_files.fileids():
	a_n =  filename[:3]
	auth_map[filename] =  [a_n]

problem_cat = CategorizedPlaintextCorpusReader(problem_root, '.*\.txt', cat_map=auth_map)

documents = [(list(problem_cat.words(fileid)), category) 
				for category in problem_cat.categories() 
				for fileid in problem_cat.fileids(category)]

random.shuffle(documents)

#print problem_cat.categories()
#for category in problem_cat.categories():
#	print problem_cat.fileids(category)