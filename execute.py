import parsers
import evaluator


# ohsumed

count_matrix, category, categories_dictionary = parsers.text_data_parser('ohscal')
auc_performance, acc_performance = evaluator.evaluate(count_matrix, category, len(categories_dictionary))
print('for the OHSUMED dataset')
print('The AUC values with no averaging and their standard deviation for the 5 sizes of training sets are:')
print(auc_performance)
print('While the accuracy values are:')
print(acc_performance)
print('/n/n')

# rcv1V2

train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('CCAT')
acc_performance = evaluator.evaluate_pre_split_no_auc(train_count_matrix, train_category, test_count_matrix,
                                                      test_category, len(categories_dictionary))
print('for the RCV1 Corporate dataset')
print('The accuracy values with their standard deviations are:')
print(acc_performance)
print('while the AUC values were not calculable due to issues')
print('/n/n')

train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('ECAT')
acc_performance = evaluator.evaluate_pre_split_no_auc(train_count_matrix, train_category, test_count_matrix,
                                                      test_category, len(categories_dictionary))
print('for the RCV1 Economics dataset')
print('The accuracy values with their standard deviations are:')
print(acc_performance)
print('while the AUC values were not calculable due to issues')
print('/n/n')

train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('GCAT')
acc_performance = evaluator.evaluate_pre_split_no_auc(train_count_matrix, train_category, test_count_matrix,
                                                      test_category, len(categories_dictionary))
print('for the RCV1 Government dataset')
print('The accuracy values with their standard deviations are:')
print(acc_performance)
print('while the AUC values were not calculable due to issues')
print('/n/n')

train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.rcv1_parser('MCAT')
acc_performance = evaluator.evaluate_pre_split_no_auc(train_count_matrix, train_category, test_count_matrix,
                                                      test_category, len(categories_dictionary))
print('for the RCV1 Market dataset')
print('The accuracy values with their standard deviations are:')
print(acc_performance)
print('while the AUC values were not calculable due to issues')
print('/n/n')

# 20news

train_count_matrix, train_category, test_count_matrix, test_category, categories_dictionary =\
    parsers.news20_parser()
acc_performance = evaluator.evaluate_pre_split_no_auc(train_count_matrix, train_category, test_count_matrix,
                                                      test_category, len(categories_dictionary))
print('for the 20news Market dataset')
print('The accuracy values with their standard deviations are:')
print(acc_performance)

