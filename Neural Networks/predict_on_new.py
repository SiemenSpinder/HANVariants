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
import tldextract

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
from models_opt import CNN_model_opt, BiGRU_model_opt, ANConv1D_model_opt, ANConv2D_model_opt, AN2Conv1D_model_opt, AN_model_opt, AN2_model_opt, AN3_model_opt
from models import CNN_model, BiGRU_model, ANConv1D_model, ANConv2D_model, AN2Conv1D_model, AN_model, AN2_model, AN3_model
from spaces import CNN_space, BiGRU_space, ANConv1D_space, ANConv2D_space, AN2Conv1D_space, AN_space, AN2_space, AN3_space

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
    SUBSET = False
    MODEL = 'CNN' #Choose from 'CNN', 'BiGRU', 'SentANConv1D', 'SentANConv2D', 'PageANConv1D', 'PageAN2Conv1D', 'DomainAN'
                    # 'DomainAN2Sent', 'DomainAN2Page', 'DomainAN3'
    EMBEDDING = 'no embedding' #Choose from  'no embedding', 'Word2vec' and 'fasttext'
    NUMB_FEAT = 30000
    PATH = r'C:\Users\sieme\Documents\CBS\senttextfiles_industrie'
    FILE_PATH = MODEL + EMBEDDING
    NB = 0
    if SUBSET == True:
        NB = 2000
    VOCAB = 'vocab.txt'
    EMBED_SIZE = 150
    PATIENCE = 0
    EPOCHS = 60
    if EMBEDDING != 'no embedding':
        pretrained = True
    else:
        pretrained = False

    if MODEL in ['CNN', 'BiGRU', 'DomainAN']:
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
                     PATH, NB, VOCAB, pretrained, EMBEDDING, EMBED_SIZE, save_path), file_pi)

    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'wb') as file_pi:
        pickle.dump((MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path), file_pi)

Xtest, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = data()

if MODEL == 'CNN':
        model = CNN_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test)
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

Adam = optimizers.Adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH)+ 'BEST' + ".hdf5", monitor='val_loss', verbose=1,
                       save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

model.load_weights(os.path.join(save_path,FILE_PATH) + 'BEST' +".hdf5")

with open(os.path.join(save_path,MODEL + EMBEDDING + '.pickle'), 'rb') as file_pi:
        history, metrics, tokenizer = pickle.load(file_pi)

y_pred = model.predict(Xtest)
temp = []
for y in y_pred:
    if y>0.5:
        temp.append(1)
    else:
        temp.append(0)
y_pred = temp

NB_PRINT = 20

for i in range(NB_PRINT):
    print('Website:', doc_files[i].replace('.txt', ''),
         ' Predicted: ',  y_pred[i])

websites_in_order = [doc_files[i].replace('.txt', '') for i in range(len(y_pred))]

df = pd.DataFrame({'domains': websites_in_order,
                   'predict': y_pred})

#finds list of URLs
df_tot = pd.read_excel(r'C:\Users\sieme\Documents\CBS\IndustrieSelectie.xlsx')
df_tot = df_tot[pd.notnull(df_tot['URL'])]
urls = df_tot['URL'].values.tolist()

#convert URLs to domains
def converttodom(url):
    tlresult = tldextract.extract(url)
    temp = tlresult.domain + '.' + tlresult.suffix
    if len(temp) >4:
        domain = temp
        return domain
    
df_tot['domains'] = df_tot['URL'].apply(converttodom)

df = df.merge(df_tot, on = 'domains')

vertaal = pd.read_excel(r'C:\Users\sieme\Documents\CBS\Neural Networks\VertaalLocatie.xlsx')

df = df.merge(vertaal, left_on = 'Cbp_VestigingsAdresPostcodeNumeriek', right_on = 'PC')

writer = pd.ExcelWriter('PreditionsWebsitesIndustrie.xlsx')
df.to_excel(writer, 'Sheet1')
writer.save()

print(datetime.now() - startTime)
