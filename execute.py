import parsers
import evaluator


def print_results(dataset_name, auc_performance, acc_performance):
    print('For the ' + dataset_name + ' dataset')
    print('The AUC values with no averaging and their standard deviation for the 4 sizes of training sets are:')
    print(auc_performance)
    print('(If N/A, it means it was not calculable due to unknown issues)')
    print('The accuracy values are:')
    print(acc_performance)
    print('\n\n')


# ohsumed

count_matrix, category, categories_dictionary = parsers.text_data_parser('ohscal')
auc_performance, acc_performance = evaluator.evaluate(count_matrix, category, len(categories_dictionary))
print_results('OHSUMED', auc_performance, acc_performance)

# rcv1V2

train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('CCAT')
auc_performance, acc_performance = evaluator.evaluate_pre_split(train_count_matrix, train_category, test_count_matrix,
                                                                test_category, len(categories_dictionary))
print_results('RCV1-V2 Corporate', auc_performance, acc_performance)


train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('ECAT')
auc_performance, acc_performance = evaluator.evaluate_pre_split(train_count_matrix, train_category, test_count_matrix,
                                                                test_category, len(categories_dictionary))
print_results('RCV1-V2 Economics', auc_performance, acc_performance)


train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('GCAT')
auc_performance, acc_performance = evaluator.evaluate_pre_split(train_count_matrix, train_category, test_count_matrix,
                                                                test_category, len(categories_dictionary))
print_results('RCV1-V2 Government', auc_performance, acc_performance)


train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('MCAT')
auc_performance, acc_performance = evaluator.evaluate_pre_split(train_count_matrix, train_category, test_count_matrix,
                                                                test_category, len(categories_dictionary))
print_results('RCV1-V2 Market', auc_performance, acc_performance)


# 20news

train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.news20_parser()
auc_performance, acc_performance = evaluator.evaluate_pre_split(train_count_matrix, train_category, test_count_matrix,
                                                                test_category, len(categories_dictionary))
print_results('20News', auc_performance, acc_performance)


