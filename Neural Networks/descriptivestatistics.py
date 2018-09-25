#basic packages
import os, re, csv, math, codecs
import itertools
from datetime import datetime
import numpy as np
from numpy import array
from numpy import zeros
import pandas as pd
import pickle

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

#scipy for some statistics
from scipy import stats

#custom function to preprocess
from processdocs import load_doc, process_docs

startTime = datetime.now()

# set paths
PATH_POS = r'C:\Users\sieme\Documents\CBS\senttextfiles'
PATH_NEG = r'C:\Users\sieme\Documents\CBS\senttextfiles2'
VOCAB = 'vocab.txt'

save_path = os.getcwd() + "\\" +'descriptivestatistics'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load the vocabulary and stopwords
vocab = load_doc(VOCAB)
vocab = vocab.split()
vocab = set(vocab)

with open(r'stopwords-nl.txt',  'r', encoding = 'utf8') as f:
    dutchstopwords = f.read().split('\n')

with open(r'custom_stopwords.txt', 'r', encoding = 'utf8') as f:
    customstopwords = f.read().split('\n')

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

positive_texts = process_docs(PATH_POS, positive_doc_files, vocab, 'DomainAN2Page', stopwords)
add_positive_texts = process_docs(PATH_NEG, positive_doc_files, vocab, 'DomainAN2Page', stopwords)
negative_texts = process_docs(PATH_NEG, negative_doc_files, vocab, 'DomainAN2Page', stopwords)
add_negative_texts = process_docs(PATH_POS, negative_doc_files, vocab, 'DomainAN2Page', stopwords)
positive_texts = positive_texts + add_positive_texts
negative_texts = negative_texts + add_negative_texts
texts = positive_texts + negative_texts

positive_texts2 = process_docs(PATH_POS, positive_doc_files, vocab, 'DomainAN2Sent', stopwords)
add_positive_texts2 = process_docs(PATH_NEG, positive_doc_files, vocab, 'DomainAN2Sent', stopwords)
negative_texts2 = process_docs(PATH_NEG, negative_doc_files, vocab, 'DomainAN2Sent', stopwords)
add_negative_texts2 = process_docs(PATH_POS, negative_doc_files, vocab, 'DomainAN2Sent', stopwords)
positive_texts2 = positive_texts2 + add_positive_texts2
negative_texts2 = negative_texts2 + add_negative_texts2
texts2 = positive_texts2 + negative_texts2

positive_texts3 = process_docs(PATH_POS, positive_doc_files, vocab, 'DomainAN3', stopwords)
add_positive_texts3 = process_docs(PATH_NEG, positive_doc_files, vocab, 'DomainAN3', stopwords)
negative_texts3 = process_docs(PATH_NEG, negative_doc_files, vocab, 'DomainAN3', stopwords)
add_negative_texts3 = process_docs(PATH_POS, negative_doc_files, vocab, 'DomainAN3', stopwords)
positive_texts3 = positive_texts3 + add_positive_texts3
negative_texts3 = negative_texts3 + add_negative_texts3
texts3 = positive_texts3 + negative_texts3

tokenizeddocs = [doc.split() for doc in docs]
lengthdocs = [len(doc) for doc in tokenizeddocs]

lengthstexts = []
numbersentences = []
for text in texts:
    lengthssents = []
    for sentence in text:
        splitsentence = sentence.split()
        lengthssents.append(len(splitsentence))
    lengthstexts.append(lengthssents)
    numbersentences.append(len(lengthssents))

sents = list(itertools.chain.from_iterable(lengthstexts))

lengthstexts2 = []
numbersentences2 = []
for text in texts2:
    lengthssents = []
    for sentence in text:
        splitsentence = sentence.split()
        lengthssents.append(len(splitsentence))
    lengthstexts2.append(lengthssents)
    numbersentences2.append(len(lengthssents))

sents2 = list(itertools.chain.from_iterable(lengthstexts2))

lengthstexts3 = []
for text in texts3:
    lengthssents = []
    for sentence in text:
        lengthssents.append(len(sentence))
    lengthstexts3.append(lengthssents)

sents3 = list(itertools.chain.from_iterable(lengthstexts3))

font = {'family' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

def CalcDescrStats(data):
    DS = stats.describe(data)
    Mean = DS.mean
    Median = np.median(data)
    StDev = math.sqrt(DS.variance)
    IQR = stats.iqr(data)
    TukeyFence = Median + 1.5*IQR
    Max = DS.minmax[1]
    print('Mean: ', Mean, 'Median: ', Median, 'StDev: ', StDev,
          'IQR: ', IQR, 'TukeyFence: ', TukeyFence, 'Max: ', Max)

print('Descriptive Stats WpS: ', CalcDescrStats(sents2))
print('Descriptive Stats WpP: ', CalcDescrStats(sents))
print('Descriptive Stats WpD: ', CalcDescrStats(lengthdocs))
print('Descriptive Stats SpP: ', CalcDescrStats(sents3))
print('Descriptive Stats SpD: ', CalcDescrStats(numbersentences2))
print('Descriptive Stats PpD: ', CalcDescrStats(numbersentences))

plt.figure(1)
bins = np.linspace(0, 4000, 40)
plt.hist(lengthdocs, bins, alpha = 0.5, histtype = 'bar',ec = 'black')
plt.title('Number of words per domain', fontsize=20)
plt.axvline(x = 2500, color = 'black')
plt.savefig(os.path.join(save_path,'numb_word_per_dom.png'))

plt.figure(2)
bins = np.linspace(0, 800, 40)
plt.hist(sents, bins, alpha = 0.5, histtype = 'bar',ec = 'black')
plt.title('Number of words per web page')
plt.axvline(x = 400, color = 'black')
plt.savefig(os.path.join(save_path,'numb_word_per_page.png'))

plt.figure(3)
bins = np.linspace(0, 15, 15)
plt.hist(numbersentences, bins, alpha = 0.5, histtype = 'bar',ec = 'black')
plt.title('Number of web pages per domain')
plt.axvline(x = 15, color = 'black')
plt.savefig(os.path.join(save_path,'numb_page_per_dom.png'))

plt.figure(4)
bins = np.linspace(0, 500, 40)
plt.hist(numbersentences2, bins, alpha = 0.5, histtype = 'bar',ec = 'black')
plt.title('Number of sentences per domain')
plt.axvline(x = 310, color = 'black')
plt.savefig(os.path.join(save_path,'numb_sent_per_dom.png'))

plt.figure(5)
bins = np.linspace(0, 40, 40)
plt.hist(sents2, bins, alpha = 0.5, histtype = 'bar',ec = 'black')
plt.title('Number of words per sentence')
plt.axvline(x = 15, color = 'black')
plt.savefig(os.path.join(save_path,'numb_word_per_sent.png'))

plt.figure(6)
bins = np.linspace(0, 150, 40)
plt.hist(sents3, bins, alpha = 0.5, histtype = 'bar',ec = 'black')
plt.title('Number of sentences per web page')
plt.axvline(x = 30, color = 'black')
plt.savefig(os.path.join(save_path,'numb_sent_per_page.png'))

         
    
