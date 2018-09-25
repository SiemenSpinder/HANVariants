from os import listdir
import os
from datetime import datetime
from nltk.tokenize import sent_tokenize

startTime = datetime.now()

MIN_PAGES = 2
MAX_PAGES = 15

def convert_to_proper_page(webpage):

    a = ['cookie', 'Cookie', 'inloggen', 'registreren', 'aanmelden' 'nieuwsbrief', 'Aanmelden', 'account', 'Inloggen',
     'instagram', 'facebook', 'wachtwoord', 'e-mailadres', 'Lees']

    text = sent_tokenize(webpage)
    text = [tex for tex in text if not any(x in tex for x in a)]
    return text

def convert_domain_to_pages(domain):
    proper_pages = [' '.join(convert_to_proper_page(page)) for page in domain]
    proper_pages = list(filter(None, proper_pages))
    return proper_pages


def preprocess(raw_text):
    #split text into web pages and concatenates everything in web page
    temp = []
    for j in raw_text.splitlines():
        temp.append(' '.join([i for i in j.split()]))
    #cleans web pages from general web info
    temp = convert_domain_to_pages(temp)
    #removes all domains with less than 2 web pages, more than 15 or if they are completely empty
    if (not len(temp) <MIN_PAGES) and (not len(temp) > MAX_PAGES):
        return "\n            ".join(temp)


print('Loading unedited text files..')

complete_path = r'C:\Users\sieme\Documents\CBS\textfiles_industrie'
textdir = listdir(complete_path)

texts = []
for document in textdir:
    with open(os.path.join(complete_path,document) , 'r', encoding = 'utf8') as text_file:
        texts.append(text_file.read())

print('Remove documents with less than ',MIN_PAGES,' pages or more than ',MAX_PAGES, ' pages plus removing sentences with certain texts')

cleantexts = [preprocess(text) for text in texts]

newpath = r'C:\Users\sieme\Documents\CBS\senttextfiles_industrie'

print('Put preprocessed files into new directory')

#make directory for .txt files
try:
        os.makedirs(newpath)
except OSError:
        pass

for index, document in enumerate(os.listdir(complete_path)):
    if cleantexts[index]:
        with open(os.path.join(newpath, document) , 'w', encoding = 'utf8') as fw:
                fw.write("%s\n" % cleantexts[index])

print(datetime.now() - startTime)

