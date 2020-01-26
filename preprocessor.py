from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from unidecode import unidecode
import re

ps = PorterStemmer()
with open('SMART stop words') as small_file:  # getting stop words for later
    stop_words = small_file.read().split('\n')
    stop_words2 = []
    for stop in stop_words:
        for non_alpha2 in re.findall('[^\w\s]', stop):
            stop = stop.replace(non_alpha2, '')
        stop_words2.append(stop)
    stop_words = stop_words2


def news20_preprocess(corpus):  # to be used only with this particular work
    if 'Lines:' in corpus:
        corpus = corpus.split('Lines', 1)[1]
    if 'Writes:' in corpus:
        corpus = corpus.split('Writes', 1)[1]
    if 'Archive-name:' in corpus and 'Last-modified:' in corpus and 'Version:' in corpus:
        corpus = corpus.split('Version', 1)[1]
    # removal of metadata that could overly help the classifier, it's a bit of a mess
    # since they all have different structures
    corpus = corpus.replace('\t', ' ')  # word_tokenize doesn't seem to process this easily
    corpus = corpus.replace('\n', ' ')
    corpus = corpus.lower()
    for non_alpha in re.findall('[^\w\s]', corpus):
        corpus = corpus.replace(non_alpha, '')
    for number in re.findall('[0-9]+', corpus):
        corpus = corpus.replace(number, '')
    corpus = unidecode(corpus)
    corpus = word_tokenize(corpus)
    corpus = [x for x in corpus if x not in stop_words]
    postprocess = ''
    for word in corpus:
        word = ps.stem(word)
        postprocess = postprocess + word + ' '
    return postprocess
