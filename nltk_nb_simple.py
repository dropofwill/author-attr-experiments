import nltk

data = ([ ([1,1,0,1,1,0], 0) ] +
		[ ([1,0,1,0,1,1], 1) ])

classifier = nltk.NaiveBayesClassifier.train(data)