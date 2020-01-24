import os

from scipy.sparse import csc_matrix

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from bs4 import BeautifulSoup

from preprocessor import rcv1_preprocess

ps = PorterStemmer()
with open('SMART stop words') as small_file:  # getting stop words for later
    words = word_tokenize(small_file.read())
    words_stemmed = []
    for word in words:
        words_stemmed.append(ps.stem(word))
        words_stemmed.append(word)
    stop_words_tuple = tuple(words_stemmed)


def text_data_parser(name):
    if name + '.mat' in os.listdir('./text-data/'):   # checking if file name is valid
        document_row = []
        document_column = []
        document_value = []
        with open('./text-data/' + name + '.mat') as documents:  # reading matrix of token counts
            dimensions = word_tokenize(documents.readline())  # could use python string's methods, but this saves me /n
            dimensions = [int(dimensions[0]), int(dimensions[1])]
            current_row = 0
            for line in documents:  # for every document
                data = word_tokenize(line)
                for i in range(int(len(data)/2)):  # for every word instance
                    document_row.append(current_row)
                    document_column.append(int(data[2*i]) - 1)  # -1 since my matrix starts at 0
                    document_value.append(int(float(data[2*i+1])))
                current_row += 1
        counts_matrix = csc_matrix((document_value, (document_row, document_column)),
                                   shape=(dimensions[0], dimensions[1]))
        categories_names = dict()
        document_category = []
        with open('./text-data/' + name + '.mat.rlabel') as categories:
            for line in categories:
                line = line.strip('\n')
                category = categories_names.get(line, -1)
                if category == -1:
                    new = len(categories_names)
                    categories_names[line] = new
                    category = new
                document_category.append(category)
        return counts_matrix, document_category, categories_names
    else:
        return 0, 0, 0


def rcv1_parser(macro_category):
    category_list = dict()
    with open('./Rcv1/rcv1-topics') as file:
        counter = 0
        for line in file:
            line = word_tokenize(line)
            if len(line) > 5 and line[2] == macro_category:
                category_list[line[5]] = counter
                counter += 1
    category = dict()  # this uses the .qrels file to get the category for each document OLDID
    with open('./Rcv1/rcv1-v2.topics.qrels') as file:
        for line in file:
            line = line.split(' ')
            if len(line) > 0 and category_list.get(line[0], 0) != 0:
                category[line[1]] = category_list[line[0]]
        all_text = []  # here we start parsing the documents
    document_category = []
    files = os.listdir('./Rcv1')   # all files in the database directory
    data_sets = []
    for file in files:
        if 'reut2-' in file:
            data_sets.append(file)
    for data_set in data_sets:
        with open('./Rcv1/' + data_set, errors='ignore') as file:  # hand to add errors='ignore', there's a corrupted
            # character somehwere todo use training set
            print('working with ' + data_set + ' now')
            soup = BeautifulSoup(file, 'html.parser')
            for document in soup.find_all('reuters'):
                if category.get(document.get('oldid'), 0) != 0:
                    body = document.find('body')
                    if body:  # some entries have no body, they will be ignored
                        corpus = document.find('title')
                        corpus = corpus.get_text()
                        corpus = corpus + ' ' + body.get_text()
                        document_category.append(category[document.get('oldid')])
                        all_text.append(corpus)
    vectorizer = CountVectorizer(stop_words=stop_words_tuple)
    count_matrix = vectorizer.fit_transform(all_text)
    return count_matrix, document_category, category_list
