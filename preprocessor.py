from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import unidecode
import re

with open('SMART stop words') as file:# getting stop words for later
    stop_words_tuple = tuple(word_tokenize(file.read()))
ps = PorterStemmer()


def rcv1_preprocess(corpus):  # to be used only with this particular work
    corpus = corpus.replace('\t', ' ')  # word_tokenize doesn't seem to process this easily
    corpus = corpus.replace('\n', ' ')
    corpus = word_tokenize(corpus)
    corpus.pop()
    corpus.pop()  # remove junk at the end that this set has
    postprocess = []
    for word in corpus:
        word = word.lower()
        word = unidecode.unidecode(word)
        for non_alpha in re.findall('[\W]', word):
            word = word.replace(non_alpha, '')
        if len(re.findall('[^0-9]', word)) != 0 and (word not in stop_words_tuple):  # removing numbers and stop words
            word = ps.stem(word)    # default porter stemmer
            postprocess.append(word)

    return postprocess
