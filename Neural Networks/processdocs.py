#basic
import os
from os import listdir
from string import punctuation

#nltk
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding = 'utf8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc, vocab, stopwords):
        stopword_set = set(stopwords)
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.lower().translate(table) for w in tokens]
        tokens = [w for w in tokens if w not in stopword_set]
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
        return tokens

def clean_doc2(doc, vocab, stopwords):
        stopword_set = set(stopwords)
        doc = doc.split('       ')  
        sents = []
        for sent in doc:
                tokens = sent.split()
                table = str.maketrans('', '', punctuation)
                tokens = [w.lower().translate(table) for w in tokens]
                tokens = [w for w in tokens if w not in stopword_set]
                tokens = [w for w in tokens if w in vocab]
                tokens = ' '.join(tokens)
                sents.append(tokens)
        return sents

def clean_doc3(doc, vocab, stopwords):
        stopword_set = set(stopwords)
        doc = sent_tokenize(doc) 
        sents = []
        for sent in doc:
                tokens = sent.split()
                table = str.maketrans('', '', punctuation)
                tokens = [w.lower().translate(table) for w in tokens]
                tokens = [w for w in tokens if w not in stopword_set]
                tokens = [w for w in tokens if w in vocab]
                tokens = ' '.join(tokens)
                sents.append(tokens)
        return sents

def clean_doc4(doc, vocab, stopwords):
        stopword_set = set(stopwords)
        doc = doc.split('       ')
        pages = []
        for page in doc:
                sentences = sent_tokenize(page)
                sents = []
                for sent in sentences:
                        tokens = sent.split()
                        table = str.maketrans('', '', punctuation)
                        tokens = [w.lower().translate(table) for w in tokens]
                        tokens = [w for w in tokens if w not in stopword_set]
                        tokens = [w for w in tokens if w in vocab]
                        tokens = ' '.join(tokens)
                        sents.append(tokens)
                pages.append(sents)
        return pages
                        
# load all docs in a directory
def process_docs(directory, files, vocab, MODEL, stopwords):
        documents = list()
        # walk through all files in the folder
        for document in files:
        # create the full path of the file to open
                path = os.path.join(directory,document)
                # load the doc
                try:
                        doc = load_doc(path)
                        # clean doc
                        if (MODEL == 'PageANConv1D' or MODEL == 'DomainAN2Page'):
                                tokens = clean_doc2(doc, vocab, stopwords)
                        elif (MODEL == 'SentANConv1D' or MODEL == 'DomainAN2Sent'):
                                tokens = clean_doc3(doc, vocab, stopwords)
                        elif MODEL in ['SentANConv2D', 'PageAN2Conv1D', 'DomainAN3']:
                                tokens = clean_doc4(doc, vocab, stopwords)
                        else:
                                tokens = clean_doc(doc, vocab, stopwords)
                        # add to list
                        documents.append(tokens)
                except:
                        pass
        return documents

