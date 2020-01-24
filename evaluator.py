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
                                                                train_size=size)  # random_state=42 + i)
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
            auc_results.append(metrics.roc_auc_score(y_true, y_score, multi_class='ovo')*100)  # for percentages
        acc_performance.append([np.mean(acc_results), np.std(acc_results)])
        auc_performance.append([np.mean(auc_results), np.std(auc_results)])
    auc_perf = []
    for pair in auc_performance:
        auc_perf.append([format(pair[0], '.4f'), format(pair[1], '.4f')])
    acc_perf = []
    for pair in acc_performance:
        acc_perf.append([format(pair[0], '.4f'), format(pair[1], '.4f')])
    return auc_perf, acc_perf
