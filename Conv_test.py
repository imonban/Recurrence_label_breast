from __future__ import print_function
import tensorflow as tf
import keras
config = tf.ConfigProto()
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout
from keras.models import Model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from keras.utils import np_utils
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pickle




            

def cleanupDoc(s):
    stopset = set(stopwords.words('english'))
    #tokens = word_tokenize(s)
    cleanup = " ".join(filter(lambda word: word not in stopset, s.split()))
    cleanup1 = " ".join([w for w in cleanup.split() if len(w)>2])
    return cleanup1

def conver_index(txt):
    tokens = txt.split()
    con_text = []
    for t in tokens:
        try:
            i = model.vocab[t].index
            con_text.append(i)
        except:
            print('word not found: '+t)
    return con_text



def classification_test(df):
    # set parameters:
    max_features = 100  # vocabulary size
    maxlen = 1000  # maximum length of the review
    batch_size = 32
    embedding_dims = 20
    ngram_filters = [3, 5, 7]
    nb_filter = 1200  # number of filters for each ngram_filter
    print('Loading data...')
    stop = stopwords.words('english') + list(string.punctuation)
    X_test = []
    for i in range(df.shape[0]):
        SENT = df.iloc[i]['TAG_SENT']
        Temp_sent = re.sub('\d', ' ', SENT)
        tokens = word_tokenize(Temp_sent)
        x = cleanupDoc(Temp_sent)
        X_test.append(x)
    with open('./model/tokenizer_tag.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)


    print('Pad sequences (samples x time)')
    x_test = sequence.pad_sequences(X_test_sequences, maxlen=maxlen)
    print('X_test shape:', x_test.shape)
    
    ## load model
    # Model reconstruction from JSON file
    with open('./model/CNNmodel_tag.json', 'r') as f:
        model = model_from_json(f.read())
    ## load weights
    # Load weights into the new model
    model.load_weights('./model/weights_tag.json.h5')
    prediction = model.predict(x_test)
    P = []
    for i in range(len(prediction)):
        if np.argmax(prediction[i]) == 1:
            P.append('Definite recurrence')
        else:
            P.append('No recurrence')
    df['Predicted'] = P
    df['Predicted'] = prediction
    df = df.drop(['NOTE', 'UN_SENT'], axis =1)
    return df
