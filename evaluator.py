import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

training_test_sizes = [64, 128, 256, 512]


def evaluate(count_matrix, category, categories_count):
    auc_performance = []
    acc_performance = []
    for size in training_test_sizes:
        auc_results = []
        acc_results = []
        for i in range(30):
            x_train, x_test, y_train, y_test = train_test_split(count_matrix, category,
                                                                train_size=size-categories_count)
            '''  later if I have to do the force, rn let's just use the pre processed data
            y_append = []
            z_append = []  # indeces of elements to remove from test matrix
            i = 0
            while len(y_append) < categories_count and i < len(y_test):
                if y_test[i] not in y_append:
                    z_append.append(i)
                    y_append.append(y_test[i])
                i += 1
            # after this we should have 1 element per category in the append set, could misbehave if all of category
            # already in training set, implausible, in that case training set will be smaller
            z_append = np.array(z_append)
            x_append = x_test[z_append]
            valids = np.arange(len(y_test))
            x_test =
            '''

            model = MultinomialNB()
            model.fit(x_train, y_train)
            classification = model.predict(x_test)
            probability_classification = model.predict_proba(x_test)
            acc_results.append(np.mean(classification == y_test)*100)  # for percentages

            # now work only for roc
            y_true = []
            for j in y_test:
                row = [0]*categories_count
                row[j] = 1
                y_true.append(row)
            y_score = np.asarray(probability_classification)
            auc_result = metrics.roc_auc_score(y_true, y_score, multi_class='ovo') * 100  # for percentages
            try:
                auc_result = metrics.roc_auc_score(y_true, y_score, multi_class='ovo')*100  # for percentages
            except ValueError:
                pass
            if auc_result != 0:
                auc_results.append(auc_result)
        acc_performance.append([np.mean(acc_results), np.std(acc_results)])
        if len(auc_results) == 0:
            auc_performance.append('N/A')
        else:
            auc_performance.append([np.mean(auc_results), np.std(auc_results)])
    auc_perf = []
    for pair in auc_performance:
        if pair == 'N/A':
            auc_perf.append('N/A')
        else:
            auc_perf.append([format(pair[0], '.4f'), format(pair[1], '.4f')])
    acc_perf = []
    for pair in acc_performance:
        acc_perf.append([format(pair[0], '.4f'), format(pair[1], '.4f')])
    return auc_perf, acc_perf


def evaluate_pre_split(train_count_matrix, train_category, test_count_matrix, test_category, categories_count):
    train_category = np.array(train_category)
    acc_performance = []
    auc_performance = []
    for size in training_test_sizes:
        acc_results = []
        auc_results = []
        for i in range(30):
            x_train, x_test, y_train, y_test = train_test_split(train_count_matrix, train_category,
                                                                train_size=size - categories_count)
            # x_test and y_test are actually ignored since sub parts of the training set
            model = MultinomialNB()
            model.fit(x_train, y_train)
            classification = model.predict(test_count_matrix)
            probability_classification = model.predict_proba(x_test)
            acc_results.append(np.mean(classification == test_category) * 100)  # for percentages
            auc_result = 0   # auc result attempt in case it works
            y_true = []
            for j in y_test:
                row = [0] * categories_count
                row[j] = 1
                y_true.append(row)
            y_score = np.asarray(probability_classification)
            try:
                auc_result = metrics.roc_auc_score(y_true, y_score, multi_class='ovo') * 100  # for percentages
            except ValueError:
                pass
            if auc_result != 0:
                auc_results.append(auc_result)
        acc_performance.append([np.mean(acc_results), np.std(acc_results)])
        if len(auc_results) == 0:
            auc_performance.append('N/A')
        else:
            auc_performance.append([np.mean(auc_results), np.std(auc_results)])

    auc_perf = []
    for pair in auc_performance:
        if pair == 'N/A':
            auc_perf.append('N/A')
        else:
            auc_perf.append([format(pair[0], '.4f'), format(pair[1], '.4f')])
    acc_perf = []
    for pair in acc_performance:
        acc_perf.append([format(pair[0], '.4f'), format(pair[1], '.4f')])
    return auc_perf, acc_perf
