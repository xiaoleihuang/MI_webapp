import numpy as np
import os
import sys

# keras preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

np.random.seed(1337)  # for reproducibility

# for word2vec model
import gensim

# save storage
import json
try:
	import cPickle as pickle
except:
	import pickle

if sys.version_info[0] == 2:
	pass

def load_init_properties(fpath='./resources/settings.ini'):
	"""Load properties of initializations
	"""
	import configparser
	if os.path.isfile(fpath):
		config = configparser.ConfigParser()
		config.read_file(open(fpath))
		return config
	else:
		return None

def padding2sequences(X, MAX_NB_WORDS=10000, MAX_SEQUENCE_LENGTH=40, tokenizer=None):
	"""Padding sentences."""
	if not tokenizer:
		if keras.__version__.startswith('2'):
			tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
		else:
			tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
		tokenizer.fit_on_texts(X)
	sequences = tokenizer.texts_to_sequences(X)
	
	print('Found %s unique tokens.' % len(tokenizer.word_index))
	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	return data, tokenizer

def chars_dict_builder(X):
    """
    Create a character 2 indices mapping.
    Preprocess step for Character level CNN classification.
    """
    chars_set = set(" ".join([sent for sent in X]))
    print('total number of chars: ', len(chars_set))

    char_indices = dict((c, i) for i, c in enumerate(chars_set))
    return char_indices

def encode2char(X, maxlen, char_indices):
    """
    One hot encoding for running character-level neural networks;
    However, this method always produce too large sparse matrix
    """
    #indices_char = dict((i, c) for i, c in enumerate(chars_dict))
    char_size = len(char_indices)

    # convert X to indices
    data = np.zeros((len(X), maxlen, char_size), dtype=np.bool)

    for index, sent in enumerate(X):
        counter = 0
        sent_array = np.zeros((maxlen, char_size))
        # list(sent.lower().replace(' ', '')) to remove all whitespaces
        # need to test which works better
        chars = list(sent.lower())

        for c in chars:
            if counter >= maxlen:
                break
            else:
                char_array = np.zeros(char_size, dtype=np.int)
                if c in char_indices:
                    ix = char_indices[c]
                    char_array[ix] = 1
                sent_array[counter, :] = char_array
                counter += 1
        data[index, :, :] = sent_array
    return data

def encode2char_bme(X, maxlen, char_indices):
    """
    One hot encoding for the sentence, but another strategy:
        each word will be modeled by begin, middle and end
    """
    char_size = len(char_indices)
    data = np.zeros((len(X), maxlen*3, char_size),dtype=np.int)

    for index, sent in enumerate(X):
        counter = 0
        sent_array = np.zeros((maxlen*3, char_size))
        words = list(sent.lower().split())

        for w in words:
            if counter >= maxlen*3:
                break
            else:
                begin_array = np.zeros(char_size, dtype=np.int)
                if w[0] in char_indices:
                    ix = char_indices[w[0]]
                    begin_array[ix] = 1
                sent_array[counter,:] = begin_array
                counter += 1

                mid_array = np.zeros(char_size, dtype=np.int)
                if len(w) > 2:
                    for c in w[1:-1]:
                        if c in char_indices:
                            ix = char_indices[c]
                            mid_array[ix] = 1
                sent_array[counter,:] = mid_array
                counter += 1

                end_array = np.zeros(char_size, dtype=np.int)
                if len(w) > 1:
                    if w[-1] in char_indices:
                        ix = char_indices[w[-1]]
                        end_array[ix] = 1
                sent_array[counter,:] = end_array
                counter += 1
        data[index, :, :] = sent_array
    return data

def mini_batch_generator(X, y, batch_size=128):
    """
    Mini batch data generator
    """
    for i in range(0, len(X), batch_size):
        x_sample = X[i : i+batch_size]
        y_sample = y[i : i+batch_size]
        yield(x_sample, y_sample)

def convert_code(code):
	if '+' in code or '-' in code:
		if int(code[-2:]) > 0:
			return 1
		else:
			return -1
	else:
		return 0

def encode_codes(codes):
	"""Encode CODES to +1, 0, -1, such as 'O+2' and 'O+3' will be +1;
	'O-3' and 'O-4' will be -1.

	Examples:

	Args:
		codes (list): list of codes

	Return:
		list of encoded codes
	"""
	return [convert_code(item) for item in codes]


def preproc_data(data, use_lower=True, use_stem=False, use_stopwords=False, split_sent=False):
	"""Functions to preprocess the dataset"""
	dataset = []

	for doc in data:
		doc = doc.strip()
		if use_lower:
			doc = doc.lower()
		
		tmp_doc = word_tokenize(doc)
		tmp_doc = [word.strip() for word in tmp_doc if len(word.strip()) > 0]
		if use_stem:
			tmp_doc = [stemmer.stem(word) for word in tmp_doc]
		if use_stopwords:
			stopwords_set = set(stopwords.words('english'))
			tmp_doc = [word for word in tmp_doc if word not in stopwords_set]
		if not split_sent:
			tmp_doc = " ".join(tmp_doc)
		dataset.append(tmp_doc)
	return dataset

def proc_pipeline(input_data, keras_tokenizer, max_len):
	# due to preprocessing APIs are handle a list of data
	if type(input_data) == str:
		input_data = unicode(input_data)
		input_data = [input_data]
	
	# convert the dataset to index
	dataset_idx = keras_tokenizer.texts_to_sequences(input_data)
	length_data = len(dataset_idx)
	dataset_idx = pad_sequences(dataset_idx, maxlen=max_len)
	return dataset_idx
