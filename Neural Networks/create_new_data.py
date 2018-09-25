#basic packages
import os, re, csv, math, codecs
import random
import pickle
import numpy as np
from numpy import array
from numpy import zeros
import pandas as pd

#sk learn packages
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#process docs
from processdocs import load_doc, process_docs

#print most informative features logit and svm
from attentionandfeatures import show_most_informative_features, most_informative_feature_for_class_svm

#keras preprocessing (for deep learning)
from keras import backend as K
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

#loading the embeddings
from processembeddings import load_embedding

def data():
    random.seed(7)
    np.random.seed(7)

    def convert_to_data(x, tokenizer, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT):
        X = np.zeros((len(x), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        for i, sentences in enumerate(x):
            for j, sent in enumerate(sentences):
                if j< MAX_SENTS:
                    wordTokens = text_to_word_sequence(sent)
                    #update 1/10/2017 - bug fixed - set max number of words
                    k=0
                    for _, word in enumerate(wordTokens):
                        if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<NUMB_FEAT:
                            X[i,j,k] = tokenizer.word_index[word]
                            k=k+1
        return X

    def convert_to_data2(x, tokenizer, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT):
        X = np.zeros((len(x), MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        for i, pages in enumerate(x):
            for j, sentences in enumerate(pages):
                if j<MAX_PAGES:
                    for k, sent in enumerate(sentences):
                        if k< MAX_SENTS:
                            wordTokens = text_to_word_sequence(sent)
                            #update 1/10/2017 - bug fixed - set max number of words
                            l=0
                            for _, word in enumerate(wordTokens):
                                if l<MAX_SENT_LENGTH and tokenizer.word_index[word]<NUMB_FEAT:
                                    X[i,j,k,l] = tokenizer.word_index[word]
                                    l=l+1
        return X

    with open(os.path.join(os.getcwd(), 'config_data' + '.pickle'), 'rb') as file_pi:
        MODEL, NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENT_LENGTH, MAX_SENTS, SUBSET, PATH, NB, VOCAB, pretrained, EMBEDDING, EMBED_SIZE, save_path = pickle.load(file_pi)

    # load the vocabulary and stopwords
    vocab = load_doc(VOCAB)
    vocab = vocab.split()
    vocab = set(vocab)

    with open(r'stopwords-nl.txt',  'r', encoding = 'utf8') as f:
        dutchstopwords = f.read().split('\n')

    with open(r'custom_stopwords.txt', 'r', encoding = 'utf8') as f:
        customstopwords = f.read().split('\n')

    print('Choosing random documents')
    doc_files = os.listdir(PATH)

    if SUBSET == True:        
        random.shuffle(doc_files)
        doc_files = doc_files[:NB]

    stopwords =  dutchstopwords + customstopwords

    # load all documents
    print('Loading documents...')
    docs = process_docs(PATH, doc_files, vocab, None, stopwords)
    
    if MODEL in ['SentANConv1D', 'SentANConv2D', 'PageANConv1D', 'PageAN2Conv1D', 'DomainAN2Sent', 'DomainAN2Page', 'DomainAN3']:
        texts = process_docs(PATH, doc_files, vocab, MODEL, stopwords)
        x_test = texts
    else:
        x_test = docs

    # create the tokenizer
    print('Tokenizing documents...')
    with open(os.path.join(save_path,MODEL + EMBEDDING + '.pickle'), 'rb') as file_pi:
                history, metrics, tokenizer = pickle.load(file_pi)
    # fit the tokenizer on the documents
    if MODEL in ['SentANConv1D', 'PageANConv1D', 'DomainAN2Sent', 'DomainAN2Page']:
        Xtest = convert_to_data(x_test, tokenizer, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
    elif MODEL in ['SentANConv2D', 'PageAN2Conv1D', 'DomainAN3']:
        Xtest = convert_to_data2(x_test, tokenizer, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
    else:
        # sequence encode
        test_encoded = tokenizer.texts_to_sequences(x_test)
        # pad sequences
        Xtest = pad_sequences(test_encoded, maxlen=MAX_LENGTH, padding='post')

    # define test labels
    print('Shape')
    print(Xtest.shape)
    if pretrained == True:
    # load embedding from file
        if EMBEDDING == 'fasttext':
            print('Loading Fasttext...')
            raw_embedding = load_embedding('wiki.nl.vec', EMBED_SIZE)
        else:
            print('Loading Word2vec..')
            raw_embedding = load_embedding('wikipedia-160.txt', EMBED_SIZE)
    else:
        raw_embedding = 0

    idx2 = 0
    return Xtest, NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test

