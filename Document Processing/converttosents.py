from os import listdir
import os
from datetime import datetime
from nltk.tokenize import sent_tokenize

startTime = datetime.now()

MIN_PAGES = 1
MAX_PAGES = 15

#Removes sentences containing certain words, these sentences are usually noise
def convert_to_proper_page(webpage):
    a = ['cookie', 'Cookie', 'inloggen', 'registreren', 'aanmelden' 'nieuwsbrief', 'Aanmelden', 'account', 'Inloggen',
     'instagram', 'facebook', 'wachtwoord', 'e-mailadres', 'Lees']
    text = sent_tokenize(webpage)
    text = [tex for tex in text if not any(x in tex for x in a)]
    return text

#Concatenates all web pages to one domain, filter empty pages
def convert_domain_to_pages(domain):
    proper_pages = [' '.join(convert_to_proper_page(page)) for page in domain]
    proper_pages = list(filter(None, proper_pages))
    return proper_pages

#Does all preprocessing, lots of space is added to make it easy to recognize where a webpage starts
def preprocess(raw_text):
    #split text into web pages and concatenates everything in web page
    temp = []
    for j in raw_text.splitlines():
        temp.append(' '.join([i for i in j.split()]))
    #cleans web pages from general web info
    temp = convert_domain_to_pages(temp)
    #removes all domains with less than MIN_PAGES web pages, more than MAX_PAGES or if they are completely empty
    if (not len(temp) <MIN_PAGES) and (not len(temp) > MAX_PAGES):
        return "\n            ".join(temp)


print('Loading unedited text files..')

complete_path = r'C:\Users\sieme\Documents\CBS\textfiles'
textdir = listdir(complete_path)

texts = []
for document in textdir:
    with open(os.path.join(complete_path,document) , 'r', encoding = 'utf8') as text_file:
        texts.append(text_file.read())


complete_path2 = r'C:\Users\sieme\Documents\CBS\textfiles2'
textdir = listdir(complete_path2)

texts2 = []
for document in textdir:
    with open(os.path.join(complete_path2,document) , 'r', encoding = 'utf8') as text_file:
        texts2.append(text_file.read())

print('Remove documents with less than ',MIN_PAGES,' pages or more than ',MAX_PAGES, ' pages plus removing sentences with certain texts')

cleantexts = [preprocess(text) for text in texts]
cleantexts2 = [preprocess(text) for text in texts2]

newpath = r'C:\Users\sieme\Documents\CBS\senttextfiles'
newpath2 = r'C:\Users\sieme\Documents\CBS\senttextfiles2'

print('Put preprocessed files into new directory')

#make directory for .txt files
try:
        os.makedirs(newpath)
except OSError:
        pass

#make directory for .txt files
try:
        os.makedirs(newpath2)
except OSError:
        pass

    
#print cleanedfiles into directory
for index, document in enumerate(os.listdir(complete_path)):
    if cleantexts[index]:
        with open(os.path.join(newpath, document) , 'w', encoding = 'utf8') as fw:
                fw.write("%s\n" % cleantexts[index])

for index, document in enumerate(os.listdir(complete_path2)):
    if cleantexts2[index]:
        with open(os.path.join(newpath2, document) , 'w', encoding = 'utf8') as fw:
                fw.write("%s\n" % cleantexts2[index])

print(datetime.now() - startTime)

