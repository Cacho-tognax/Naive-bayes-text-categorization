from preprocessor import rcv1_preprocess
import parsers
import evaluator


# ohsumed first since it's the first one to actually work

count_matrix, category, categories_dictionary = parsers.text_data_parser('ohscal')
auc_performance, acc_performance = evaluator.evaluate(count_matrix, category, len(categories_dictionary))
print('for the OHSUMED dataset')
print('The AUC values with no averaging and their standard deviation for the 5 sizes of training sets are:')
print(auc_performance)
print('While the accuracy values are:')
print(acc_performance)
print('/n/n')

# that was easy, now rcv-1

count_matrix, category, categories_dictionary = parsers.rcv1_parser('CCAT')
auc_performance, acc_performance = evaluator.evaluate(count_matrix, category, len(categories_dictionary))
print('for the RCV1 dataset')
print('The AUC values with no averaging and their standard deviation for the 5 sizes of training sets are:')
print(auc_performance)
print('While the accuracy values are:')
print(acc_performance)
print('/n/n')
