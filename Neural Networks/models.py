#basic
import os, re, csv, math, codecs
import pickle

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

def CNN_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
    
##    numb_filt1 = 64
##    kernel_size1 = 3
##    pooling_size1 = 3
##    weight_decay1 = 0
##    dropout1 = 0.21
##    
##    numb_filt2 = 32
##    kernel_size2 = 3
##    pooling_size2 = 5
##    weight_decay2 = 0
##    dropout2 = 0.33

    numb_filt1 = 32
    kernel_size1 = 3
    pooling_size1 = 7
    weight_decay1 = 0
    dropout1 = 0.171
        
    numb_filt2 = 64
    kernel_size2 = 7
    pooling_size2 = 7
    weight_decay2 = 0
    dropout2 = 0.540

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
        
    model.add(Conv1D(numb_filt1, kernel_size1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay1)))
    model.add(MaxPooling1D(pooling_size1))
    model.add(Dropout(dropout1))

    model.add(Conv1D(numb_filt2, kernel_size2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay2)))
    model.add(MaxPooling1D(pooling_size2))
    model.add(Dropout(dropout2))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)
    return model

def CNN_1_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    dropout0 = 0
    
    numb_filt1 = 64
    kernel_size1 = 5
    pooling_size1 = 5
    weight_decay1 = 0
    dropout1 = 0.599

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
        
    model.add(Conv1D(numb_filt1, kernel_size1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay1)))
    model.add(MaxPooling1D(pooling_size1))
    model.add(Dropout(dropout1))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)
    return model

def CNN_3_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    dropout0 = 0
    
    numb_filt1 = 32
    kernel_size1 = 7
    pooling_size1 = 5
    weight_decay1 = 0
    dropout1 = 0.410
    
    numb_filt2 = 32
    kernel_size2 = 3
    pooling_size2 = 5
    weight_decay2 = 0
    dropout2 = 0.144

    numb_filt3 = 32
    kernel_size3 = 5
    pooling_size3 = 3
    weight_decay3 = 0
    dropout3 = 0.041

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
        
    model.add(Conv1D(numb_filt1, kernel_size1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay1)))
    model.add(MaxPooling1D(pooling_size1))
    model.add(Dropout(dropout1))

    model.add(Conv1D(numb_filt2, kernel_size2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay2)))
    model.add(MaxPooling1D(pooling_size2))
    model.add(Dropout(dropout2))

    model.add(Conv1D(numb_filt3, kernel_size3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay3)))
    model.add(MaxPooling1D(pooling_size3))
    model.add(Dropout(dropout3))
    model.add(Flatten())
    
    model.add(Dense(1, activation='sigmoid'))  #multi-label (k-hot encoding)
    return model

def BiGRU_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    gruunits = 64
    grurecdrop1 = 0.3
    
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
        
    model.add(Bidirectional(GRU(gruunits, recurrent_dropout=grurecdrop1)))
    
    model.add(Dense(1, activation='sigmoid'))
    return model

def ANConv1D_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    if MODEL == 'SentANConv1D':
        gruunits1 = 128
        grurecdrop1 = 0.066

        numb_filt1 = 32
        kernel_size1 = 5
        weight_decay1 = 0
        pooling_size1 = 5
        dropout1 = 0.046

    if MODEL == 'PageANConv1D':
        gruunits1 = 32
        grurecdrop1 = 0.171

        numb_filt1 = 32
        kernel_size1 = 3
        weight_decay1 = 0
        pooling_size1 = 3
        dropout1 = 0.387
    
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
    l_lstm = Bidirectional(GRU(gruunits1, return_sequences = True , recurrent_dropout=grurecdrop1))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    x = Conv1D(numb_filt1, kernel_size1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay1))(review_encoder)
    x = MaxPooling1D(pooling_size1)(x)
    x = Dropout(dropout1)(x)
    x = Flatten()(x)
    
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(review_input, preds)
    return (model, sentEncoder)

def ANConv2D_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
    
    gruunits1 = 64
    grurecdrop1 = 0.073

    numb_filt1 = 128
    kernel_size1 = 5
    weight_decay1 = 0
    pooling_size1 = 3
    dropout1 = 0.049

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
    l_lstm = Bidirectional(GRU(gruunits1, return_sequences = True , recurrent_dropout=grurecdrop1))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    emptydistr = Model(review_input, review_encoder)

    total_input = Input(shape=(MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH), dtype = 'int32')
    x = TimeDistributed(emptydistr)(total_input)
    x = Conv2D(numb_filt1, kernel_size1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay1))(x)
    x = MaxPooling2D(pooling_size1)(x)
    x = Dropout(dropout1)(x)
    x = Flatten()(x)

    preds = Dense(1, activation='sigmoid')(x)
    model = Model(total_input, preds)
    return (model, sentEncoder)

def AN2Conv1D_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
    
    gruunits1 = 128
    grurecdrop1 = 0.271

    gruunits2 = 128
    grurecdrop2 = 0.262

    numb_filt1 = 64
    kernel_size1 = 7
    weight_decay1 = 0
    pooling_size1 = 3
    dropout1 = 0.331

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
    l_lstm = Bidirectional(GRU(gruunits1, return_sequences = True , recurrent_dropout=grurecdrop1))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    webpage_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    webpage_encoder = TimeDistributed(sentEncoder)(webpage_input)
    l_lstm_sent = Bidirectional(GRU(gruunits2, return_sequences = True,  recurrent_dropout=grurecdrop2))(webpage_encoder)
    attention_sent = AttentionWithContext()(l_lstm_sent)
    webpageEncoder = Model(webpage_input, attention_sent)

    domain_input = Input(shape=(MAX_PAGES,MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(webpageEncoder)(domain_input) 
    x = Conv1D(numb_filt1, kernel_size1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay1))(review_encoder)
    x = MaxPooling1D(pooling_size1)(x)
    x = Dropout(dropout1)(x)
    x = Flatten()(x)
    
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(domain_input, preds)
    return (model, webpageEncoder, sentEncoder)


def AN_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
    
    gruunits = 64
    grurecdrop1 = 0.003
    
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
        
    model.add(Bidirectional(GRU(gruunits, return_sequences=True, recurrent_dropout=grurecdrop1)))
    model.add(AttentionWithContext())
    
    model.add(Dense(1, activation='sigmoid'))
    return model

def AN2_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)

    if MODEL == 'DomainAN2Sent':
        gruunits1 = 32
        grurecdrop1 = 0.120
        
        gruunits2 = 128
        grurecdrop2 = 0.045

    if MODEL == 'DomainAN2Page':
        gruunits1 = 64
        grurecdrop1 = 0.001
        
        gruunits2 = 32
        grurecdrop2 = 0.066 
    
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
    
    l_lstm = Bidirectional(GRU(gruunits1, return_sequences = True , recurrent_dropout=grurecdrop1))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(gruunits2, return_sequences = True,  recurrent_dropout=grurecdrop2))(review_encoder)
    attention_sent = AttentionWithContext()(l_lstm_sent)
    
    preds = Dense(1, activation='sigmoid')(attention_sent)
    model = Model(review_input, preds)
    return (model, sentEncoder)

def AN3_model(NUMB_FEAT, MAX_LENGTH, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH, tokenizer, raw_embedding, doc_files, idx2, x_test):
    with open(os.path.join(os.getcwd(), 'config_model' + '.pickle'), 'rb') as file_pi:
        MODEL, pretrained, EMBEDDING, EMBED_SIZE, PATIENCE, EPOCHS, FILE_PATH, save_path, train_emb = pickle.load(file_pi)
    
    gruunits1 = 128
    grurecdrop1 = 0.495
    
    gruunits2 = 128
    grurecdrop2 = 0.286

    gruunits3 = 32
    grurecdrop3 = 0.237

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
    
    l_lstm = Bidirectional(GRU(gruunits1, return_sequences = True , recurrent_dropout=grurecdrop1))(embedded_sequences)
    attention = AttentionWithContext()(l_lstm)
    sentEncoder = Model(sentence_input, attention)

    webpage_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    webpage_encoder = TimeDistributed(sentEncoder)(webpage_input)
    l_lstm_sent = Bidirectional(GRU(gruunits2, return_sequences = True,  recurrent_dropout=grurecdrop2))(webpage_encoder)
    attention_sent = AttentionWithContext()(l_lstm_sent)
    webpageEncoder = Model(webpage_input, attention_sent)

    domain_input = Input(shape=(MAX_PAGES,MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(webpageEncoder)(domain_input)
    l_lstm_webpage = Bidirectional(GRU(gruunits3, return_sequences = True,  recurrent_dropout=grurecdrop3))(review_encoder)
    attention_webpage = AttentionWithContext()(l_lstm_webpage)

    preds = Dense(1, activation='sigmoid')(attention_webpage)
    model = Model(domain_input, preds)
    return (model, webpageEncoder, sentEncoder)
