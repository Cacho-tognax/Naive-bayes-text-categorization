#Naive bayes text categorization


This code requires specific files from specific datasets in specific folders to run, to be precise:
1. The OSHUMED dataset must be provided in the  preprocessed form from 'tmdata.tar.gz' 
available [here](https://matplotlib.org/) in a folder called text-data.
1. The 20 Newsgroup dataset available [here](http://qwone.com/~jason/20Newsgroups/) has to be the bydate version, 
the 2 folders from that archive in the root folder of the project.
1. For the Rcv1-V2 dataset the files from the appendixes 
1, 2, 12.i from [here](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/) must
 be provided in a folder called Rcv1V2
1. Additionally the SMART stop words list available in the appendix 11 from the previous link must be provided as 
'SMART stop words' in the root folder of the project.
 
 Running execute.py once the data is provided will yield pairs of [average, standard deviation] from 3 tests for 
 both the AUC and accuracy scores for 4 sample sizes [64, 128, 256, 512] in the terminal.
 

#Libraries used

This code uses the [numpy](https://numpy.org/), [nltk](https://www.nltk.org/), [scipy](https://www.scipy.org/),
 [sklearn](https://www.scipy.org/) and [unidecode](https://pypi.org/project/Unidecode/) libraries.
 Additionally [matplotlib](https://matplotlib.org/) has been used to make the graphs in the PDF relation.
 
#Citations
 
 Rcv1-V2 dataset :
 
 Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361-397, 2004. http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf
 
 nltk :
 
 Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.