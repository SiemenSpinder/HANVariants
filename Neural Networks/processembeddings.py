import numpy as np
from numpy import asarray
from numpy import zeros

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab, NUMB_FEAT, EMBED_SIZE):
	# total vocabulary size plus 0 for unknown words
	vocab_size = NUMB_FEAT +1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, EMBED_SIZE))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		if i >= NUMB_FEAT:
			continue
		embedding_vector = embedding.get(word)
		if embedding_vector is not None:
			weight_matrix[i] = embedding_vector
	return weight_matrix

def load_embedding(filename, EMBED_SIZE):
        embedding = {}
        f = open(filename, 'r', encoding = 'utf8')
        for line in f:
                values = line.split()
                word = ' '.join(values[:-EMBED_SIZE])
                coefs = np.asarray(values[-EMBED_SIZE:], dtype='float32')
                embedding[word] = coefs.reshape(-1)
        return embedding
        f.close()
