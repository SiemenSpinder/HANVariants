#basic
import os
from os import listdir
import numpy as np

#nltk
from nltk.tokenize import word_tokenize

#gensim
import gensim
from gensim.models import FastText

print('Load documents...')
complete_path = r'C:\Users\sieme\Documents\CBS\senttextfiles'
textdir = listdir(complete_path)

texts = []
for document in textdir:
    with open(os.path.join(complete_path,document) , 'r', encoding = 'utf8') as text_file:
        texts.append(text_file.read())

complete_path2 = r'C:\Users\sieme\Documents\CBS\senttextfiles2'
textdir = listdir(complete_path2)

texts2 = []
for document in textdir:
    with open(os.path.join(complete_path2,document) , 'r', encoding = 'utf8') as text_file:
        texts2.append(text_file.read())

completetexts = texts + texts2
print('Total training websites: %d' % len(completetexts))

#tokenize all documents
tokenizedtexts = [word_tokenize(text) for text in completetexts]

#create Word2vec embeddings
model = gensim.models.Word2Vec(tokenizedtexts, size = 150,
                               window = 10, min_count = 2, workers =5)
model.train(tokenizedtexts , total_examples=len(tokenizedtexts ), epochs=10)

#print to file
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
