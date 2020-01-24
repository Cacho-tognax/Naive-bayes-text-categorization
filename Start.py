import fileinput
import numpy.random as rng
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
file = open('SMART stop words')
stop_words_tuple = tuple(word_tokenize(file.read()))

