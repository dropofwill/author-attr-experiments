import time
import os

# List of function words from http://www.flesl.net/Vocabulary/Single-word_Lists/function_word_list.php

function_words = ['about','across', 'against','along','around','at','behind',
	'beside','besides','by','despite','down','during','for','from','in','inside','into','near','of','off',
	'on','onto','over','through','to','toward','with','within','without','I','you','he','me','her','him','my','mine',
	'her','hers','his','myself','himself','herself','anything','everything','anyone','everyone','ones',
	'such','it','we','they','us','them','our','ours','their','theirs','itself','ourselves','themselves','something',
	'nothing','someone','the','some','this','that','every','all','both','one','first','other','next','many','much',
	'more','most','several','no','a','an','any','each','half','twice','two','second','another','last','few','little','less',
	'least','own','and','but','after','when','as','because','if','what','where','which','how','than','or','so','before','since',
	'while','although','though','who','whose','can','may','will','shall','could','might','would','should','must','be',
	'do','have','here','there','today','tomorrow','now','then','always','never','sometimes','usually','often','therefore',
	'however','besides','moreover','though','otherwise','else','instead','anyway','incidentally','meanwhile']

twain_quotes = ("Apparently there is nothing that cannot happen today.", "If you tell the truth, you don't have to remember anything.")
hemmingway_quotes = ("Happiness in intelligent people is the rarest thing I know.", "Courage is grace under pressure.")

sub_dir = "Results/"
location = "results" + time.strftime("%Y%m%d-%H%M%S") + ".txt"

with open( os.path.join(sub_dir, location), 'w+') as myFile:
    myFile.write("yoloswag")