#basic
import os
from os import listdir
import pandas
import tldextract

#text extraction
from bs4 import BeautifulSoup
import justext

Justtext = False

#path of HTML files
complete_path = r"C:\Users\sieme\Documents\CBS\NonSustain\FullHTML\htmloutput"
htmldir = listdir(complete_path)

# function takes HTML and converts to text
def pathtotext(joinedpath):
	# ensures right decoding is used
    decoded = False
    for cp in ('utf-8','utf8','cp850', 'cp1252'):
        try:
            with open(joinedpath, encoding=cp) as fp:
                if Justtext == False:
                    soup = BeautifulSoup(fp,'lxml')
                else:
                    text = fp.read()
                decoded = True
                break
        except:
            pass
            
    if decoded:
        try:
            if Justtext == True:
                paragraphs = justext.justext(text, justext.get_stoplist("Dutch"))
                text = ' '.join([paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate])
            else:
                for script in soup(["script", "style"]):
                    script.decompose()    # rip it out

                # get text
                text = soup.get_text(" ")

                # break into lines and remove leading and trailing space on each
                lines = (line.strip() for line in text.splitlines())
                # break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # drop blank lines
                text = ' '.join(chunk for chunk in chunks if chunk)
            text = text.replace('\n', ' ')
            return(text)
        except:
            return('')
            pass

#finds paths to all HTML files
paths = []
for k, html in enumerate(htmldir):
    joinedpath = os.path.join(complete_path, htmldir[k])
    paths.append(joinedpath)


#finds list of URLs
df = pandas.read_excel('NotSustainList.xlsx')
df = df[pandas.notnull(df['URL'])]
urls = df['URL'].values.tolist()

#convert URLs to domains
domains = []
for url in urls:
    try:
        tlresult = tldextract.extract(url)
        domain = tlresult.domain + '.' + tlresult.suffix
        if len(domain) >4:
            domains.append(domain)
    except:
        pass

def clean_path(path):
    path = path.split('\\')[-1]
    if 'www.' in path:
        path = path.split('www.')[1]
    if '.nl' in path:
        path = path.split('.nl')[0] + '.nl'
        return path
    elif '.com' in path:
        path = path.split('.com')[0] + '.com'
        return path
    elif '.org' in path:
        path = path.split('.org')[0] + '.org'
        return path
    else:
        pass
    
#path to save .txt files to
newpath = r'C:\Users\sieme\Documents\CBS\textfiles2' 

#make directory for .txt files
try:
        os.makedirs(newpath)
except OSError:
        pass

for s in paths:
    cleans = clean_path(s)
    if cleans in set(domains):
        temp = pathtotext(s)
        if temp is not None:
            temp = temp + '\n'
            if temp != '\n':
                with open(os.path.join(newpath, cleans) + '.txt' , 'a', encoding = 'utf8') as text_file:
                    text_file.write(temp)
    
