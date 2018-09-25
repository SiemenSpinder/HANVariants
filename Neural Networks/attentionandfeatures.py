from keras import backend as K
import numpy as np

def show_most_informative_features(vectorizer, ch2, clf, n=20):
    feature_names = vectorizer.get_feature_names()
        # keep selected feature names
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-25s\t\t%.4f\t%-25s" % (coef_1, fn_1, coef_2, fn_2))

def most_informative_feature_for_class_svm(vectorizer, ch2,  classifier,  classlabel, n=10):
    labelid = classlabel # this is the coef we're interested in. 
    feature_names = vectorizer.get_feature_names()
    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    svm_coef = classifier.coef_.toarray() 
    topn = sorted(zip(svm_coef[labelid], feature_names), reverse = True)[0:n]

    for coef, feat in topn:
        print(feat, coef)   

def get_attention(model, sequence_input, attentionlayer):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[attentionlayer-1].output])
    out = get_layer_output([sequence_input, 0])[0]
    uit = np.tanh(np.dot(out[0], model.layers[attentionlayer].get_weights()[0]) + model.layers[attentionlayer].get_weights()[1])
    ait = np.dot(uit, model.layers[attentionlayer].get_weights()[2])
    a = np.exp(ait)
    a = a/sum(a)
    return a

def print_attention_AN_inp1_words(model, Xtest, file_nb, sent_nb, tokenizer, MAX_LENGTH):
    word_attention = get_attention(model, Xtest[file_nb].reshape(1,MAX_LENGTH), 2)

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    words = [reverse_word_map[i] for i in Xtest[file_nb] if i != 0]
    
    words_attentionlist = list(zip(words, word_attention.tolist()))

    return(words_attentionlist)

def print_attention_AN_inp2_words(model, Xtest, file_nb, sent_nb, tokenizer, MAX_SENT_LENGTH):
    word_attention = get_attention(sentEncoder, Xtest[file_nb][sent_nb].reshape(1,MAX_SENT_LENGTH), 3)

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    words = [reverse_word_map[i] for i in Xtest[file_nb][sent_nb] if i != 0]
    
    words_attentionlist = list(zip(words, word_attention.tolist()))
    
    return(words_attentionlist)

def print_attention_AN_inp3_words(model, Xtest, file_nb, page_nb, sent_nb, tokenizer, MAX_SENT_LENGTH):
    word_attention = get_attention(model, Xtest[file_nb][page_nb][sent_nb].reshape(1,MAX_SENT_LENGTH), 3)

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    words = [reverse_word_map[i] for i in Xtest[file_nb][page_nb][sent_nb] if i != 0]
    
    words_attentionlist = list(zip(words, word_attention.tolist()))
    
    return(words_attentionlist)

def print_attention_AN2_inp2_sents(model, Xtest, x_test, file_nb, tokenizer, MAX_SENTS, MAX_SENT_LENGTH):
    sent_attention = get_attention(model, Xtest[file_nb].reshape(1, MAX_SENTS, MAX_SENT_LENGTH), 3)
    
    sent_attentionlist = list(zip(x_test[file_nb], sent_attention.tolist()))
    
    return(sent_attentionlist)

def print_attention_AN2_inp3_sents(model, Xtest, x_test, file_nb, sent_nb, tokenizer, MAX_SENTS, MAX_SENT_LENGTH):
    sent_attention = get_attention(model, Xtest[file_nb][sent_nb].reshape(1, MAX_SENTS, MAX_SENT_LENGTH), 3)
    
    sent_attentionlist = list(zip(x_test[file_nb][sent_nb], sent_attention.tolist()))
    
    return(sent_attentionlist)

def print_attention_AN3_inp3_sents(model, Xtest, x_test, file_nb, tokenizer, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH):
    sent_attention = get_attention(model, Xtest[file_nb].reshape(1, MAX_PAGES, MAX_SENTS, MAX_SENT_LENGTH), 3)
    
    sent_attentionlist = list(zip(x_test[file_nb], sent_attention.tolist()))
    
    return(sent_attentionlist)
