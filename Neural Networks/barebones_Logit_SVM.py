#basic packages
import os, re, csv, math, codecs
import random
import pickle
import numpy as np
from numpy import array
from numpy import zeros
import pandas as pd

#sk learn packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import LogisticRegression
from sklearn import svm

#function to process documents
from processdocs import load_doc, process_docs

#functions to show most informative features
from attentionandfeatures import show_most_informative_features, most_informative_feature_for_class_svm

random.seed(7)
np.random.seed(7)

Logit = False
SVM = True
SUBSET = False
NB_OF_POS = 0
NB_OF_NEG = 0
if SUBSET == True:
    NB_OF_POS = 500
    NB_OF_NEG = 500
VOCAB = 'vocab.txt'
PATH_POS = r'C:\Users\sieme\Documents\CBS\senttextfiles'
PATH_NEG = r'C:\Users\sieme\Documents\CBS\senttextfiles2'

# load the vocabulary and stopwords
vocab = load_doc(VOCAB)
vocab = vocab.split()
vocab = set(vocab)

with open(r'stopwords-nl.txt',  'r', encoding = 'utf8') as f:
    dutchstopwords = f.read().split('\n')

with open(r'custom_stopwords.txt', 'r', encoding = 'utf8') as f:
    customstopwords = f.read().split('\n')

print('Choosing random documents')
positive_doc_files = os.listdir(PATH_POS)
negative_doc_files = os.listdir(PATH_NEG)

files_in_both = [file for file in positive_doc_files if file in negative_doc_files]

if files_in_both:
    for file in files_in_both:
        os.remove(os.path.join(PATH_POS, file))

path_neg = r'C:\Users\sieme\Documents\CBS\NotSustainList.xlsx'
path_pos = r'C:\Users\sieme\Documents\CBS\SustainList.xlsx'

df_not = pd.read_excel(path_neg)
df_yes = pd.read_excel(path_pos)

df_neg_not = df_not.loc[df_not['Judgement'] <= 0]
df_pos_not = df_not.loc[df_not['Judgement'] >= 4]
negative_files_not = df_neg_not['URL'].tolist()
positive_files_not = df_pos_not['URL'].tolist()

df_neg_yes = df_yes.loc[df_yes['Judgement'] <= 0]
df_pos_yes = df_yes.loc[df_yes['Judgement'] >= 3]
negative_files_yes = df_neg_yes['URL'].tolist()
positive_files_yes = df_pos_yes['URL'].tolist()

negative_files_not = [file + '.txt' for file in negative_files_not]
positive_files_not = [file + '.txt' for file in positive_files_not]

negative_files_yes = [file + '.txt' for file in negative_files_yes]
positive_files_yes = [file + '.txt' for file in positive_files_yes]

negative_doc_files = [negative_doc_file for negative_doc_file in negative_doc_files if negative_doc_file in negative_files_not]
add_pos_files = [positive_doc_file for positive_doc_file in negative_doc_files if positive_doc_file in positive_files_not]

positive_doc_files = [positive_doc_file for positive_doc_file in positive_doc_files if positive_doc_file in positive_files_yes]
add_neg_files = [negative_doc_file for negative_doc_file in positive_doc_files  if negative_doc_file in negative_files_yes]

positive_doc_files = positive_doc_files + add_pos_files
negative_doc_files = negative_doc_files + add_neg_files

if SUBSET == True:        
    random.shuffle(positive_doc_files)
    positive_doc_files = positive_doc_files[:NB_OF_POS]
    random.shuffle(negative_doc_files)
    negative_doc_files = negative_doc_files[:NB_OF_NEG]

doc_files = positive_doc_files + negative_doc_files

stopwords =  dutchstopwords + customstopwords

# load all documents
print('Loading documents...')
positive_docs = process_docs(PATH_POS, positive_doc_files, vocab, None, stopwords)
add_positive_docs = process_docs(PATH_NEG, positive_doc_files, vocab, None, stopwords)
negative_docs = process_docs(PATH_NEG, negative_doc_files, vocab, None, stopwords)
add_negative_docs = process_docs(PATH_POS, negative_doc_files, vocab, None, stopwords)
positive_docs = positive_docs + add_positive_docs
negative_docs = negative_docs + add_negative_docs
docs = positive_docs + negative_docs

print('Amount of positive cases: ', len(positive_docs))
print('Amount of negative cases: ', len(negative_docs))
print('Percentage positive: ', len(positive_docs)/(len(positive_docs)+len(negative_docs))*100)
print('Number of documents: ',len(docs))

indices = np.arange(0, len(docs))
target = np.concatenate([np.ones(len(positive_docs)), np.zeros(len(negative_docs))])
x_train, x_test, y_train, y_test, idx1, idx2 = train_test_split(docs, target, indices, test_size=0.2, random_state = 7)

if Logit == True or SVM == True:
    tfidf_vectorizer = TfidfVectorizer(analyzer = 'word',
                                       stop_words = [],
                                       token_pattern = r'\b[a-zA-Z]\w+\b',
                                       max_df = 0.4,
                                       min_df = 0.05,
                                       ngram_range = (1,2))
    x_train_trans = tfidf_vectorizer.fit_transform(x_train)
    x_test_trans =  tfidf_vectorizer.transform(x_test)       
    terms = tfidf_vectorizer.get_feature_names()

if Logit == True:
    ch2 = SelectKBest(chi2, k=1000)
    x_train_trans = ch2.fit_transform(x_train_trans, y_train)
    x_test_trans = ch2.transform(x_test_trans)
    clf_lr = LogisticRegression()
    clf_lr.fit(x_train_trans, y_train)
    y_pred = clf_lr.predict(x_test_trans)
    lr_score = accuracy_score(y_test, y_pred)
    precision, recall, fscore, support = score(y_test, y_pred, average = 'weighted', pos_label = 1)
    show_most_informative_features(tfidf_vectorizer, ch2, clf_lr, n =100)
    print('Test Accuracy: %f' % (lr_score*100))
    print('Test Precision: {}'.format(precision * 100))
    print('Test Recall: {}'.format(recall * 100))
    print('Test F1-score: {}'.format(fscore * 100))
    
if SVM == True:
    ch2 = SelectKBest(chi2, k=1000)
    x_train_trans = ch2.fit_transform(x_train_trans, y_train)
    x_test_trans = ch2.transform(x_test_trans)
    clf = svm.SVC(probability=True, kernel='linear')
    clf.fit(x_train_trans, y_train)
    y_pred_svc = clf.predict(x_test_trans)
    svc_score = accuracy_score(y_test, y_pred_svc)
    precision, recall, fscore, support = score(y_test, y_pred_svc, average = 'weighted', pos_label = 1)
    most_informative_feature_for_class_svm(tfidf_vectorizer, ch2, clf, 0 , n =100)
    print('Test Accuracy: %f' % (svc_score*100))
    print('Test Precision: {}'.format(precision * 100))
    print('Test Recall: {}'.format(recall * 100))
    print('Test F1-score: {}'.format(fscore * 100))
