from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import unidecode
import re

file = open('SMART stop words')  # getting stop words for later
stop_words_tuple = tuple(word_tokenize(file.read()))
file.close()
ps = PorterStemmer()


def rcv1_preprocess(tokenized):  # to be used only with this particular work
    tokenized.pop()
    tokenized.pop()  # remove junk at the end that this set has
    postprocess = []
    for word in tokenized:
        word = word.lower()
        word = unidecode.unidecode(word)
        for non_alpha in re.findall('[\W]', word):
            word = word.replace(non_alpha, '')
        if len(re.findall('[^0-9]', word)) != 0 and (word not in stop_words_tuple):  # removing numbers and stop words
            word = ps.stem(word)    # todo should I use another stemmer
            postprocess.append(word)

    return postprocess
