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


def rcv1_parser_old(macro_category):  # we have to redo all this, I was using the old dataset.
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


def rcv1_parser(macro_category):
    category_list = dict()
    with open('./Rcv1V2/rcv1-V2.topics.hier') as file:
        counter = 0
        for line in file:
            line = word_tokenize(line)
            if len(line) > 5 and line[2] == macro_category:
                category_list[line[5]] = counter
                counter += 1
    category = dict()  # this uses the .qrels file to get the category for each document OLDID
    with open('./Rcv1V2/rcv1-v2.topics.qrels') as file:
        for line in file:
            line = line.split(' ')
            if len(line) > 0 and category_list.get(line[0], 0) != 0:
                category[line[1]] = category_list[line[0]]
    all_trains = []  # here we start parsing the documents
    all_tests = []
    train_document_category = []
    test_document_category = []
    current_document = ''
    files = os.listdir('./Rcv1V2')  # all files in the database directory
    train_data_sets = []
    for file in files:
        if 'train' in file:
            train_data_sets.append(file)
    test_data_sets = []
    for file in files:
        if 'test' in file:
            test_data_sets.append(file)
    for data_set in train_data_sets:
        tex, cat = parse_lyrl(data_set, category)
        all_trains.extend(tex)
        train_document_category.extend(cat)
    for data_set in test_data_sets:
        tex, cat = parse_lyrl(data_set, category)
        all_tests.extend(tex)
        test_document_category.extend(cat)
    vectorizer = CountVectorizer()
    train_count_matrix = vectorizer.fit_transform(all_trains)
    test_count_matrix = vectorizer.transform(all_tests)
    return train_count_matrix, train_document_category, test_count_matrix, test_document_category, category_list


def parse_lyrl(file_name, category):
    texts = []
    categories = []
    with open('./Rcv1V2/' + file_name) as file:
        mode = 0  # 0 looking for next document, 1 waiting for .W, 2 reading text
        for line in file:
            line = line.rstrip()
            if mode == 0:
                if '.I' in line:  # document found
                    identifier = line.split(' ')[1]
                    if category.get(identifier, 0) != 0:  # relevant document found
                        mode = 1
                        categories.append(category[identifier])
            elif mode == 1:
                if '.W' in line:
                    mode = 2
                    current_document = ''
            elif mode == 2:
                if len(line) > 0:
                    current_document = current_document + ' ' + line
                else:
                    texts.append(current_document)
                    mode = 0
    return texts, categories
