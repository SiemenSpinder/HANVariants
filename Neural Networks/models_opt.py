#basic
import os, re, csv, math, codecs
import pickle

#hyperopt
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials

#keras basic
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint

#keras layers
from keras.layers import Conv1D, Dense, Activation, BatchNormalization, Flatten
from keras.layers import Input, TimeDistributed, MaxPooling1D,  GlobalMaxPooling1D
from keras.layers import Conv2D, MaxPooling2D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional

#load embeddings and attention layer
from processembeddings import load_embedding, get_weight_matrix
from attentionlayer import dot_product, AttentionWithContext


def CNN_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
        
    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []

    model = Sequential()
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_LENGTH, trainable=train_emb)
        model.add(embedding_layer)
    else:
        model.add(Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_LENGTH))
        
    model.add(Conv1D(params['numb_filt1'], params['kernel_size1'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0)))
    model.add(MaxPooling1D(params['pooling_size1']))
    model.add(Dropout(params['dropout1']))

    model.add(Conv1D(params['numb_filt2'], params['kernel_size2'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0)))
    model.add(MaxPooling1D(params['pooling_size2']))
    model.add(Dropout(params['dropout2']))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH)+ ".hdf5", monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH)+ ".hdf5")
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)
    
    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19}\n".format('Loss: ', val_loss,
        'Numb Filter 1: ',params['numb_filt1'], 'Numb Filter 2: ',params['numb_filt2'], 'Kernel Size 1: ',params['kernel_size1'], 'Kernel Size 2: ',params['kernel_size2'],
         'Pooling Size 1: ', params['pooling_size1'],
        'Pooling Size 2: ', params['pooling_size2'], 'Dropout 1 ', params['dropout1'], 'Dropout 2: ', params['dropout2'],
        'Learning Rate: ', params['lr']))
        
    print('Test Accuracy: %f' % (test_acc*100))
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def CNN_1_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
        
    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []

    model = Sequential()
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_LENGTH, trainable=train_emb)
        model.add(embedding_layer)
    else:
        model.add(Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_LENGTH))
        
    model.add(Conv1D(params['numb_filt1'], params['kernel_size1'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0)))
    model.add(MaxPooling1D(params['pooling_size1']))
    model.add(Dropout(params['dropout1']))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH)+ ".hdf5", monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH)+ ".hdf5")
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)
    
    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n".format('Loss: ', val_loss,
        'Numb Filter 1: ',params['numb_filt1'], 'Kernel Size 1: ',params['kernel_size1'],  'Pooling Size 1: ', params['pooling_size1'],
          'Dropout 1 ', params['dropout1'], 'Learning Rate: ', params['lr']))       
    print('Test Accuracy: %f' % (test_acc*100))
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def CNN_3_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
        
    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []

    model = Sequential()
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_LENGTH, trainable=train_emb)
        model.add(embedding_layer)
    else:
        model.add(Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_LENGTH))
        
    model.add(Conv1D(params['numb_filt1'], params['kernel_size1'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0)))
    model.add(MaxPooling1D(params['pooling_size1']))
    model.add(Dropout(params['dropout1']))

    model.add(Conv1D(params['numb_filt2'], params['kernel_size2'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0)))
    model.add(MaxPooling1D(params['pooling_size2']))
    model.add(Dropout(params['dropout2']))

    model.add(Conv1D(params['numb_filt3'], params['kernel_size3'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0)))
    model.add(MaxPooling1D(params['pooling_size3']))
    model.add(Dropout(params['dropout3']))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH)+ ".hdf5", monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH)+ ".hdf5")
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)
    
    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27}\n".format('Loss: ', val_loss,
        'Numb Filter 1: ',params['numb_filt1'], 'Numb Filter 2: ',params['numb_filt2'], 'Numb Filter 3: ',params['numb_filt3'],
         'Kernel Size 1: ',params['kernel_size1'], 'Kernel Size 2: ',params['kernel_size2'],
        'Kernel Size 3: ',params['kernel_size3'], 'Pooling Size 1: ', params['pooling_size1'], 'Pooling Size 2: ', params['pooling_size2'],
        'Pooling Size 3: ', params['pooling_size3'], 'Dropout 1 ', params['dropout1'], 'Dropout 2: ', params['dropout2'],
        'Dropout 3: ', params['dropout3'], 'Learning Rate: ', params['lr']))
        
    print('Test Accuracy: %f' % (test_acc*100))
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}


def BiGRU_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []
    
    model = Sequential()
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_LENGTH, trainable=train_emb)
        model.add(embedding_layer)
    else:
        model.add(Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_LENGTH))
        
    model.add(Bidirectional(GRU(params['gruunits'], recurrent_dropout=params['grurecdrop1'])))
    
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH))
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)

    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format('Loss: ', val_loss, 'GRU Units: ',params['gruunits'],
            'GRU Rec Drop: ', params['grurecdrop1'], 'Learning Rate: ', params['lr']))
        
    print('Test Accuracy: %f' % (test_acc*100))
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def ANConv1D_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []
    
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_SENT_LENGTH, trainable=train_emb)
    else:
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_SENT_LENGTH)
        
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(params['gruunits'], return_sequences = True , recurrent_dropout=params['grurecdrop1']))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    x = Conv1D(params['numb_filt1'], params['kernel_size1'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0))(review_encoder)
    x = MaxPooling1D(params['pooling_size1'])(x)
    x = Dropout(params['dropout1'])(x)
    x = Flatten()(x)
    
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(review_input, preds)
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH))
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)

    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}\n".format('Loss: ', val_loss,
        'GRU Units 1: ', params['gruunits'], 'Gru Rec Drop 1: ', params['grurecdrop1'], 'Number Filters 1: ',params['numb_filt1'],
        'Kernel Size 1: ',params['kernel_size1'], 'Pooling Size 1: ', params['pooling_size1'],
        'Dropout 1', params['dropout1'], 'Learning Rate: ', params['lr']))
        
    print('Test Accuracy: %f' % (test_acc*100))
    
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def ANConv2D_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []
    
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_SENT_LENGTH, trainable=train_emb)
    else:
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_SENT_LENGTH)
        
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(params['gruunits'], return_sequences = True , recurrent_dropout=params['grurecdrop1']))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    emptydistr = Model(review_input, review_encoder)

    total_input = Input(shape=(MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH), dtype = 'int32')
    x = TimeDistributed(emptydistr)(total_input)
    x = Conv2D(params['numb_filt1'], params['kernel_size1'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0))(x)
    x = MaxPooling2D(params['pooling_size1'])(x)
    x = Dropout(params['dropout1'])(x)
    x = Flatten()(x)

    preds = Dense(1, activation='sigmoid')(x)
    model = Model(total_input, preds)
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH))
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)

    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}\n".format('Loss: ', val_loss,
        'GRU Units 1: ',params['gruunits'], 'Gru Rec Drop 1: ', params['grurecdrop1'], 'Number Filters 1: ',params['numb_filt1'], 'Kernel Size 1: ',params['kernel_size1'],
        'Pooling Size 1: ', params['pooling_size1'], 'Dropout 1', params['dropout1'], 'Learning Rate: ', params['lr']))
    print('Test Accuracy: %f' % (test_acc*100))
    
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def AN2Conv1D_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
    
    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []

    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_SENT_LENGTH, trainable=train_emb)
    else:
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_SENT_LENGTH)
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(params['gruunits1'], return_sequences = True , recurrent_dropout=params['grurecdrop1']))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    webpage_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    webpage_encoder = TimeDistributed(sentEncoder)(webpage_input)
    l_lstm_sent = Bidirectional(GRU(params['gruunits2'], return_sequences = True,  recurrent_dropout=params['grurecdrop2']))(webpage_encoder)
    attention_sent = AttentionWithContext()(l_lstm_sent)
    webpageEncoder = Model(webpage_input, attention_sent)

    domain_input = Input(shape=(MAX_PAGES,MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(webpageEncoder)(domain_input) 
    x = Conv1D(params['numb_filt1'], params['kernel_size1'], activation='relu', padding='same', kernel_regularizer=regularizers.l2(0))(review_encoder)
    x = MaxPooling1D(params['pooling_size1'])(x)
    x = Dropout(params['dropout1'])(x)
    x = Flatten()(x)
    
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(domain_input, preds)
    
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH))
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)

    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19}\n".format('Loss: ', val_loss,
        'GRU Units 1: ',params['gruunits1'], 'Gru Rec Drop 1: ', params['grurecdrop1'], 'GRU Units 2: ',params['gruunits2'], 'Gru Rec Drop 2: ', params['grurecdrop2'],
      'Number Filters 1: ',params['numb_filt1'], 'Kernel Size 1: ',params['kernel_size1'], 'Pooling Size 1: ', params['pooling_size1'],
        'Dropout 1', params['dropout1'], 'Learning Rate: ', params['lr']))
    print('Test Accuracy: %f' % (test_acc*100))
    
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def AN_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []
    
    model = Sequential()
    if pretrained == True: 
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_LENGTH, trainable=train_emb)
        model.add(embedding_layer)
    else:
        model.add(Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_LENGTH))

    model.add(Bidirectional(GRU(params['gruunits'], return_sequences=True, recurrent_dropout=params['grurecdrop1'])))
    model.add(AttentionWithContext())
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH))
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)
    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format('Loss: ', val_loss, 'GRU Units: ',params['gruunits'],
            'GRU Rec Drop: ', params['grurecdrop1'], 'Learning Rate: ', params['lr']))
    print('Test Accuracy: %f' % (test_acc*100))
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def AN2_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []
    
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_SENT_LENGTH, trainable=train_emb)
    else:
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_SENT_LENGTH)
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    
    l_lstm = Bidirectional(GRU(params['gruunits1'], return_sequences = True , recurrent_dropout=params['grurecdrop1']))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(params['gruunits2'], return_sequences = True,  recurrent_dropout=params['grurecdrop2']))(review_encoder)
    attention_sent = AttentionWithContext()(l_lstm_sent)
    
    preds = Dense(1, activation='sigmoid')(attention_sent)
    model = Model(review_input, preds)
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH))
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)
    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n".format('Loss: ', val_loss, 'GRU Units 1: ',params['gruunits1'],
            'GRU Rec Drop 1: ', params['grurecdrop1'], 'GRU Units 2: ',params['gruunits2'], 'GRU Rec Drop 2: ', params['grurecdrop2'], 'Learning Rate: ', params['lr']))
    
    print('Test Accuracy: %f' % (test_acc*100))
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}

def AN3_model_opt(params):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    Xtrain, Xval, Xtest, y_train, y_val, y_test, NUMB_FEAT, MAX_LENGTH, MAX_PAGES,MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test = pickle.load(open('input.pickle','rb'))

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics = {}
    metrics['test_loss'] = []
    metrics['test_acc'] = []
    
    if pretrained == True:
        # get vectors in the right order
        embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, NUMB_FEAT, EMBED_SIZE)
        # create the embedding layer
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, weights=[embedding_vectors],
                                    input_length=MAX_SENT_LENGTH, trainable=train_emb)
    else:
        embedding_layer = Embedding(NUMB_FEAT + 1, EMBED_SIZE, input_length=MAX_SENT_LENGTH)
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    
    l_lstm = Bidirectional(GRU(params['gruunits1'], return_sequences = True , recurrent_dropout=params['grurecdrop1']))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    webpage_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    webpage_encoder = TimeDistributed(sentEncoder)(webpage_input)
    l_lstm_sent = Bidirectional(GRU(params['gruunits2'], return_sequences = True,  recurrent_dropout=params['grurecdrop2']))(webpage_encoder)
    attention_sent = AttentionWithContext()(l_lstm_sent)
    webpageEncoder = Model(webpage_input, attention_sent)

    domain_input = Input(shape=(MAX_PAGES,MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(webpageEncoder)(domain_input)
    l_lstm_webpage = Bidirectional(GRU(params['gruunits3'], return_sequences = True,  recurrent_dropout=params['grurecdrop3']))(review_encoder)
    attention_webpage = AttentionWithContext()(l_lstm_webpage)

    preds = Dense(1, activation='sigmoid')(attention_webpage)
    model = Model(domain_input, preds)
    
    print(model.summary())
    Adam = optimizers.Adam(params['lr'])
    model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    ckpt = ModelCheckpoint(os.path.join(save_path,FILE_PATH), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    model.fit(Xtrain, y_train, epochs=EPOCHS, verbose=2, batch_size = 8,
                        validation_data=(Xval, y_val), callbacks = [ckpt, early, TestCallback((Xtest, y_test))])
    model.load_weights(os.path.join(save_path,FILE_PATH))
    test_loss, test_acc = model.evaluate(Xtest, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(Xval, y_val, verbose=0)
    with open(os.path.join(save_path,FILE_PATH) + ".csv",'a') as file:
        file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}, {12}, {13}, {14}, {15}\n".format('Loss: ', val_loss, 'GRU Units 1: ',params['gruunits1'],
            'GRU Rec Drop 1: ', params['grurecdrop1'], 'GRU Units 2: ',params['gruunits2'], 'GRU Rec Drop 2: ', params['grurecdrop2'], 'GRU Units 3: ', params['gruunits3'],
            'GRU Rec Drop 3: ', params['grurecdrop3'], 'Learning Rate: ', params['lr']))
    print('Test Accuracy: %f' % (test_acc*100))
    if K.backend() == 'tensorflow':
        K.clear_session()
    return {'loss': val_loss, 'status':STATUS_OK}


