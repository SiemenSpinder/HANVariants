#basic packages
import os, re, csv, math, codecs
import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_fscore_support as score
from datetime import datetime
import pandas as pd
import pickle

#hyperopt (to optimize the parameters)
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from functools import partial

#keras packages
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint

#functions to make data
from createdata import data
#from processdocs import load_doc, clean_doc, process_docs

#model functions
from models_opt import CNN_model_opt, CNN_1_model_opt, CNN_3_model_opt, BiGRU_model_opt
from models_opt import ANConv1D_model_opt, ANConv2D_model_opt, AN2Conv1D_model_opt, AN_model_opt, AN2_model_opt, AN3_model_opt
from models import CNN_model, CNN_1_model, CNN_3_model, BiGRU_model, ANConv1D_model, ANConv2D_model, AN2Conv1D_model, AN_model, AN2_model, AN3_model
from spaces import CNN_space, CNN_1_space, CNN_3_space, BiGRU_space, ANConv1D_space, ANConv2D_space, AN2Conv1D_space, AN_space, AN2_space, AN3_space

#functions related to obtaining attention weights and logit/svm features
from attentionandfeatures import get_attention, print_attention_AN_inp1_words, print_attention_AN_inp2_words, print_attention_AN_inp3_words
from attentionandfeatures import print_attention_AN2_inp2_sents, print_attention_AN2_inp3_sents
from attentionandfeatures import print_attention_AN3_inp3_sents
from attentionandfeatures import show_most_informative_features, most_informative_feature_for_class_svm

startTime = datetime.now()


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(round(loss,4), round(acc,4)))
        metrics['test_loss'].append(loss)
        metrics['test_acc'].append(acc)
        return metrics

if __name__ == '__main__':
    TRAIN = True
    OPTIMIZE = True
    SUBSET = False
    MODEL = 'DomainAN2Sent'  #Choose from 'CNN', 'CNN_1', 'CNN_3', 'BiGRU', 'SentANConv1D', 'SentANConv2D', 'PageANConv1D', 'PageAN2Conv1D', 'DomainAN'
                    # 'DomainAN2Sent', 'DomainAN2Page', 'DomainAN3'
    EMBEDDING = 'Word2vec' #Choose from  'no embedding', 'Word2vec' and 'fasttext'
    SELForOTHER = 'Other'
    train_emb = True
    
    if train_emb == True:
        EMB = 'Further_Train'
    else:
        EMB = ''
    NUMB_FEAT = 30000
    PATH_POS = r'C:\Users\sieme\Documents\CBS\senttextfiles'
    PATH_NEG = r'C:\Users\sieme\Documents\CBS\senttextfiles2'
    FILE_PATH = MODEL + EMBEDDING + SELForOTHER + EMB
    NB_OF_POS = 0
    NB_OF_NEG = 0
    if SUBSET == True:
        NB_OF_POS = 500
        NB_OF_NEG = 500
    VOCAB = 'vocab.txt'
    if SELForOTHER == 'Self':
        EMBED_SIZE = 150
    else:
        EMBED_SIZE = 160
    PATIENCE = 0
    EPOCHS = 60
    if EMBEDDING != 'no embedding':
        pretrained = True
    else:
        pretrained = False

#obtain right dimensions for model 
    if MODEL in ['CNN', 'CNN_1', 'CNN_3', 'BiGRU', 'DomainAN']:
        MAX_LENGTH = 2500
        MAX_PAGES = 0
        MAX_SENT_LENGTH = 0
        MAX_SENTS = 0
    if MODEL in ['SentANConv1D', 'DomainAN2Sent']:
        MAX_LENGTH = 0
        MAX_PAGES = 0
        MAX_SENT_LENGTH = 20
        MAX_SENTS = 310
    if MODEL in ['PageANConv1D', 'DomainAN2Page']:
        MAX_LENGTH = 0
        MAX_PAGES = 0
        MAX_SENT_LENGTH = 400
        MAX_SENTS = 15
    if MODEL in ['SentANConv2D', 'PageAN2Conv1D', 'DomainAN3']:
        MAX_LENGTH = 0
        MAX_PAGES = 15
        MAX_SENT_LENGTH = 15
        MAX_SENTS = 30

    BEST_FILE_PATH = MODEL + EMBEDDING + 'BEST' + ".hdf5"
    save_path = os.getcwd() + "\\" + MODEL + EMBEDDING

    with open(os.path.join(os.getcwd(), 'config_data' + '.pickle'), 'wb') as file_pi:
        pickle.dump((MODEL, NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENT_LENGTH, MAX_SENTS, SUBSET,
                     PATH_POS, PATH_NEG, NB_OF_POS, NB_OF_NEG, VOCAB, pretrained, EMBEDDING, EMBED_SIZE, SELForOTHER), file_pi)

    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'wb') as file_pi:
        pickle.dump((MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb), file_pi)

if OPTIMIZE == True:
    if MODEL == 'CNN':
            used_model = CNN_model_opt
            space = CNN_space()
            
    if MODEL == 'CNN_1':
        used_model = CNN_1_model_opt
        space = CNN_1_space()
        
    if MODEL == 'CNN_3':
        used_model = CNN_3_model_opt
        space = CNN_3_space()
        
    if MODEL == 'BiGRU':
            used_model = BiGRU_model_opt
            space = BiGRU_space()

    if (MODEL == 'SentANConv1D' or MODEL == 'PageANConv1D'):
           used_model = ANConv1D_model_opt
           space = ANConv1D_space()

    if MODEL == 'SentANConv2D':
        used_model = ANConv2D_model_opt
        space = ANConv2D_space()

    if MODEL == 'PageAN2Conv1D':
        used_model = AN2Conv1D_model_opt
        space = AN2Conv1D_space()

    if MODEL == 'DomainAN':
        used_model = AN_model_opt
        space = AN_space()

    if (MODEL == 'DomainAN2Sent' or MODEL == 'DomainAN2Page'):
        used_model = AN2_model_opt
        space = AN2_space()

    if MODEL == 'DomainAN3':
        used_model = AN3_model_opt
        space = AN3_space()

    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = data()
    pickle.dump((Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test), open('input.pickle', 'wb'))

    algo = partial(tpe.suggest, n_startup_jobs=10)
    trials_step = 1
    trials_len = 1 - trials_step
    max_trials = 1
    seed = 123
    while True:
        seed = seed + 1
        try:
            trials = pickle.load(open(os.path.join(save_path, 'trials' + '.pickle'), 'rb'))
            max_trials = trials_len + trials_step
        except:
            trials = Trials()

        best =    fmin(used_model,
                  space= space,
                  algo=algo,
                  max_evals=max_trials,
                  trials=trials,
                  rstate= np.random.RandomState(seed))

        trials_len = trials_len + trials_step
        pickle.dump(trials, open(os.path.join(save_path, 'trials' + '.pickle'), 'wb'))


if OPTIMIZE == False:
    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = data()

    if MODEL == 'CNN':
            model = CNN_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)
            
    if MODEL == 'CNN_1':
            model = CNN_1_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    if MODEL == 'CNN_3':
        model = CNN_3_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)
            
    if MODEL == 'BiGRU':
            model = BiGRU_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    if (MODEL == 'SentANConv1D' or MODEL == 'PageANConv1D'):
           model, sentEncoder = ANConv1D_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    if MODEL == 'SentANConv2D':
        model, sentEncoder = ANConv2D_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    if MODEL == 'PageAN2Conv1D':
        model, webpageEncoder, sentEncoder = AN2Conv1D_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    if MODEL == 'DomainAN':
        model = AN_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    if (MODEL == 'DomainAN2Sent' or MODEL == 'DomainAN2Page'):
        model, sentEncoder = AN2_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    if MODEL == 'DomainAN3':
        model, webpageEncoder, sentEncoder = AN3_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)

    print(model.summary())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []

    LearnRate = {#'CNN': 0.0001,
                 'CNN': 0.00056,
                 'CNN_1': 0.00087,
                 'CNN_3': 0.00097,
                 'SentANConv1D' : 0.00015,
                 'PageANConv1D' : 0.00036,
                 'SentANConv2D' : 0.00035,
                 'PageAN2Conv1D' : 0.00024,
                 'DomainAN' : 0.00023,
                 'DomainAN2Sent' : 0.00070,
                 'DomainAN2Page' : 0.00014,
                 'DomainAN3' : 0.00020}

    lr = LearnRate[MODEL]
    Adam = optimizers.Adam(lr)
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH)+ 'BEST' + 'Test' + ".hdf5", monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

    if TRAIN == True:
        history = model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                            validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
        with open(os.path.join(save_path,FILE_PATH  +'.pickle'), 'wb') as file_pi:
                pickle.dump((history.history, metrics, tokenizer), file_pi)
        history = history.history
        model.load_weights(os.path.join(save_path,FILE_PATH) + 'BEST' + 'Test'  +".hdf5")
    else:
        model.load_weights(os.path.join(save_path,FILE_PATH) + 'BEST' + 'Test' +".hdf5")

        with open(os.path.join(save_path,FILE_PATH + '.pickle'), 'rb') as file_pi:
                history, metrics, tokenizer = pickle.load(file_pi)

    loss, acc = model.evaluate(Xtest, y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))

    if MODEL in ['CNN', 'CNN_1', 'CNN_3', 'BiGRU', 'DomainAN']:
        y_pred = model.predict_classes(Xtest)
    else:
        y_pred = model.predict(Xtest)
        y_pred = np.round(y_pred, decimals = 0)

    precision, recall, fscore, support = score(y_test, y_pred, average = 'weighted', pos_label = 1)

    print('Test Precision: {}'.format(precision * 100))
    print('Test Recall: {}'.format(recall * 100))
    print('Test F1-score: {}'.format(fscore * 100))

    NB_PRINT = 20

    for i in range(NB_PRINT):
        print('Website:', doc_files[idx2[i]].replace('.txt', ''),
              ' Actual: ',  y_test[i],  ' Predicted: ',  np.asscalar(y_pred[i]))

    websites_in_order = [doc_files[idx2[i]].replace('.txt', '') for i in range(len(y_test))]

    predictions = {}

    for i, website in enumerate(websites_in_order):
        predictions[website] = (y_test[i], np.asscalar(y_pred[i]))


    df = pd.DataFrame({'url': websites_in_order,
                       'predict': y_test})

    print(datetime.now() - startTime)

    #summarize history for accuracy
    MAX_X = len(history['acc'])-PATIENCE - 1
    plt.figure(1)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(list(range(1,MAX_X + 1)),history['acc'][:MAX_X])
    plt.plot(list(range(1,MAX_X + 1)),history['val_acc'][:MAX_X])
    plt.plot(list(range(1,MAX_X + 1)),metrics['test_acc'][:MAX_X])
    plt.title(MODEL + ' with ' + EMBEDDING + ', accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'test'], loc='upper left')
    plt.savefig(os.path.join(save_path,MODEL + '_' + EMBEDDING + '_' + 'accuracy.png'))

    # summarize history for loss
    plt.figure(2)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(list(range(1,MAX_X + 1)),history['loss'][:MAX_X])
    plt.plot(list(range(1,MAX_X + 1)),history['val_loss'][:MAX_X])
    plt.plot(list(range(1,MAX_X + 1)),metrics['test_loss'][:MAX_X])
    plt.title(MODEL + ' with ' + EMBEDDING + ', loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'test'], loc='upper left')
    plt.savefig(os.path.join(save_path,MODEL + '_' + EMBEDDING + '_' + 'loss.png'))
    plt.show()


