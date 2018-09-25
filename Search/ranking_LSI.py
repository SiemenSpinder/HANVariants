#standard
from os import listdir
import os, sys
from datetime import datetime
startTime = datetime.now()
import pandas as pd
import numpy as np

#stemmer
from nltk.stem.snowball import SnowballStemmer

#gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity


stemmer = SnowballStemmer("dutch")
print('Tekstbestanden laden...')

complete_path = r'C:\Users\sieme\Documents\CBS\Milieurekeningen\SitesDos\Sitespart'
textdir = listdir(complete_path)

texts = []
for document in textdir[0:10000]:
    with open(os.path.join(complete_path,document) , 'r', encoding = 'utf8') as text_file:
        texts.append((document, text_file.read()))

print('Excelbestand met andere info laden...')

path = r'C:\Users\sieme\Documents\CBS\Milieurekeningen\SitesDos\VulWebPDASv2b.xlsx'
df = pd.read_excel(path)
prompt = 'Vul keywords in om op te zoeken, geef S om te stoppen: \n'
data = []
while True:
    if data:
        A = " ".join(input().split())
    else:
        A = " ".join(input(prompt).split())    
    if len(A) == 1 and A[0].lower() == "s":
        break        
    data.append(A)
queryinput = " ".join(data)
numb_comp = input('Hoeveel bedrijven moeten er uit komen?: ')

#apply snowballstemmer
stemmedquery = " ".join([stemmer.stem(query) for query in queryinput.split()])

#remove .txt from every text file
sitenames = []
texts2 = []
for k,v in texts:
    sitenames.append(k.replace('.txt', ''))
    texts2.append(' '.join(v.split('\n')))

completetexts = [stemmedquery] + texts2
sitenames = ['0'] + sitenames

#create doc2bow corpus
dictionary = Dictionary([text.split() for text in completetexts])
dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=40000)
corpus_gensim = [dictionary.doc2bow(text.split()) for text in completetexts]

tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]
lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf] )

vec_lsi = lsi[corpus_tfidf[0]]
sims = lsi_index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
sims = sims[0:int(numb_comp)]
sims, scores = zip(*sims)
listsims = [sitenames[i] for i in sims]

new_df = pd.DataFrame({'ID': listsims[1::], 'Scores': scores[1::]})
new_df['ID'] = pd.to_numeric(new_df['ID'])

#merge highest ranked websites with information from Excel file and sort by score (descending)
df = df.merge(new_df, on = 'ID')
df = df.sort_values(by=['Scores'], ascending = False).reset_index(drop=True)

#print to new Excel file
df.to_excel('Query_LSI_test.xlsx', index=False)

print(datetime.now() - startTime)

