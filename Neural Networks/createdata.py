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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import svm

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
                            l=0
                            for _, word in enumerate(wordTokens):
                                if l<MAX_SENT_LENGTH and tokenizer.word_index[word]<NUMB_FEAT:
                                    X[i,j,k,l] = tokenizer.word_index[word]
                                    l=l+1
        return X

    Logit = False
    SVM = False
    
    with open(os.path.join(os.getcwd(), 'config_data' + '.pickle'), 'rb') as file_pi:
        MODEL, NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENT_LENGTH, MAX_SENTS, SUBSET, PATH_POS, PATH_NEG, \
        NB_OF_POS, NB_OF_NEG, VOCAB, pretrained, EMBEDDING, EMBED_SIZE, SELForOTHER = pickle.load(file_pi)

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
    if MODEL in ['SentANConv1D', 'SentANConv2D', 'PageANConv1D', 'PageAN2Conv1D', 'DomainAN2Sent', 'DomainAN2Page', 'DomainAN3']:
        positive_texts = process_docs(PATH_POS, positive_doc_files, vocab, MODEL, stopwords)
        add_positive_texts = process_docs(PATH_NEG, positive_doc_files, vocab, MODEL, stopwords)
        negative_texts = process_docs(PATH_NEG, negative_doc_files, vocab, MODEL, stopwords)
        add_negative_texts = process_docs(PATH_POS, negative_doc_files, vocab, MODEL, stopwords)
        positive_texts = positive_texts + add_positive_texts
        negative_texts = negative_texts + add_negative_texts
        texts = positive_texts + negative_texts

    print('Number of documents: ',len(docs))
    indices = np.arange(0, len(docs))
    target = np.concatenate([np.ones(len(positive_docs)), np.zeros(len(negative_docs))])
    if MODEL in ['SentANConv1D', 'SentANConv2D', 'PageANConv1D', 'PageAN2Conv1D', 'DomainAN2Sent', 'DomainAN2Page', 'DomainAN3']:
        x_train, x_test, y_train, y_test, idx1, idx2 = train_test_split(texts, target, indices, test_size=0.2, random_state = 7)
        x_train_sub, x_test_sub, y_train_sub, y_test_sub, idx1_sub, idx2_sub = train_test_split(docs, target, indices, test_size=0.2, random_state = 7)
    else:
        x_train, x_test, y_train, y_test, idx1, idx2 = train_test_split(docs, target, indices, test_size=0.2, random_state = 7)

    if Logit == True or SVM == True:
        tfidf_vectorizer = TfidfVectorizer(analyzer = 'word',
                                           stop_words = [],
                                           token_pattern = r'\b[a-zA-Z]\w+\b',
                                           max_df = 0.4,
                                           min_df = 0.05,
                                           ngram_range = (1,2))
        if MODEL in ['CNN', 'CNN_1', 'CNN_3','BiGRU', 'DomainAN']:
            x_train_trans = tfidf_vectorizer.fit_transform(x_train)
            x_test_trans =  tfidf_vectorizer.transform(x_test) 
        else:
            x_train_trans = tfidf_vectorizer.fit_transform(x_train_sub)
            x_test_trans =  tfidf_vectorizer.transform(x_test_sub)       
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
        clf = svm.SVC(probability=True, kernel='linear')
        clf.fit(x_train_trans, y_train)
        y_pred_svc = clf.predict(x_test_trans)
        svc_score = accuracy_score(y_test, y_pred_svc)
        precision, recall, fscore, support = score(y_test, y_pred_svc, average = 'weighted', pos_label = 1)
        most_informative_feature_for_class_svm(tfidf_vectorizer, clf, 0, n =100)
        print('Test Accuracy: %f' % (svc_score*100))
        print('Test Precision: {}'.format(precision * 100))
        print('Test Recall: {}'.format(recall * 100))
        print('Test F1-score: {}'.format(fscore * 100))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)

    # create the tokenizer
    print('Tokenizing documents...')
    tokenizer = Tokenizer(num_words = NUMB_FEAT)
    tokenizer.fit_on_texts(docs)
    # fit the tokenizer on the documents
    if MODEL in ['SentANConv1D', 'PageANConv1D', 'DomainAN2Sent', 'DomainAN2Page']:
        Xtrain = convert_to_data(x_train, tokenizer, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
        Xval = convert_to_data(x_val, tokenizer, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
        Xtest = convert_to_data(x_test, tokenizer, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
    elif MODEL in ['SentANConv2D', 'PageAN2Conv1D', 'DomainAN3']:
        Xtrain = convert_to_data2(x_train, tokenizer, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
        Xval = convert_to_data2(x_val, tokenizer, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
        Xtest = convert_to_data2(x_test, tokenizer, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, NUMB_FEAT)
    else:
        # sequence encode
        train_encoded = tokenizer.texts_to_sequences(x_train)
        val_encoded = tokenizer.texts_to_sequences(x_val)
        test_encoded = tokenizer.texts_to_sequences(x_test)
        # pad sequences
        Xtrain = pad_sequences(train_encoded, maxlen=MAX_LENGTH, padding='post')
        Xval = pad_sequences(val_encoded, maxlen=MAX_LENGTH, padding='post')
        Xtest = pad_sequences(test_encoded, maxlen=MAX_LENGTH, padding='post')

    # define test labels
    print('Respectively train, val and test shape')
    print(Xtrain.shape)
    print(Xval.shape)
    print(Xtest.shape)
    if pretrained == True:
    # load embedding from file
        if EMBEDDING == 'fasttext':
            print('Loading Fasttext...')
            raw_embedding = load_embedding('embedding_fasttext.txt', EMBED_SIZE)
        else:
            print('Loading Word2vec..')
            if SELForOTHER == 'Self':
                raw_embedding = load_embedding('embedding_word2vec.txt', EMBED_SIZE)
            else:
                raw_embedding = load_embedding('wikipedia-160.txt', EMBED_SIZE)
    else:
        raw_embedding = 0
    return Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test

    
