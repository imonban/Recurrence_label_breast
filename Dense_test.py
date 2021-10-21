from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding,Dropout
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



def classification_dense(df):
    # set parameters:
    vocab_size = 1200
    print('Loading data...')
    stop = stopwords.words('english') + list(string.punctuation)
    X_test = []
    for i in range(df.shape[0]):
        SENT = df.iloc[i]['TAG_SENT']
        Temp_sent = re.sub('\d', ' ', SENT)
        x = cleanupDoc(Temp_sent)
        X_test.append(x)
    with open('./model/tokenizer_dense.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    x_test = tokenizer.texts_to_matrix(X_test, mode='binary')
    
    ## load model
    # Model reconstruction from JSON file
    with open('./model/densemodel_tag.json', 'r') as f:
        model = model_from_json(f.read())
    ## load weights
    # Load weights into the new model
    model.load_weights('./model/denseweights_tag.json.h5')
    prediction = model.predict(x_test)
    P = []
    for i in range(len(prediction)):
        if prediction[i] > 0.04:
            #print(prediction[i])
            P.append(prediction[i])
        else:
            P.append(prediction[i])
    df['Predicted'] = P
    df = df.drop(['NOTE', 'UN_SENT'], axis =1)
    return df
