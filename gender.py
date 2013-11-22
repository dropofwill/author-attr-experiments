import nltk
from nltk.corpus import names
import random
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])

random.shuffle(names)

def gender_features(name):
	last_letter = name[-1]

featuresets = [(gender_features(n),g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print classifier.classify(gender_features('John'))

classifier.show_most_informative_features(25)