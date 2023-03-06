from gensim.models import KeyedVectors
from datetime import datetime

startTime = datetime.now()

#self word2vec
##model_wv = KeyedVectors.load_word2vec_format('embedding_word2vec.txt', binary = False)

#self fasttext
model_wv = KeyedVectors.load_word2vec_format('embedding_fasttext.txt', binary = False)

#other word2vec
##model_wv = KeyedVectors.load_word2vec_format('wikipedia-160.txt', binary = False)


print(datetime.now() - startTime)

