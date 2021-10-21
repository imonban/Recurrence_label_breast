from __future__ import print_function
import pandas as pd
import numpy as np
from Preprocessing import clean

import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
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
from Conv_test import classification_test
from Dense_test import classification_dense
import sys

## system argument checking
'''
if len(sys.argv) < 2:
    print('Enter directory')
    sys.exit()

directory = sys.argv[1]
'''
directory = 'C:/Users/imonb/Box/Banerjee-20190326-anonids'
def encoding(note):
    updated_note = note.encode("ascii", "ignore")
    return updated_note


def report_read(df):
    df.rename(columns={'REPORT': 'NOTE'}, inplace=True)
    df.rename(columns={'ORDERING_DATE': 'NOTE_DATE'}, inplace=True)
    df = df.fillna('N/A')
    df = df[df['NOTE']!='N/A'].reset_index(drop =  True)
    df['NOTE'] = df['NOTE'].apply(encoding)
    df['NOTE'] = df['NOTE'].astype('str')
    return df

print('Loading data....')

ids = pd.read_csv(directory +'/' +'onco_banerjee_anonids.csv')
ids = list(ids.ANON_ID)
## read files
## oncology -  progress note
df3 = pd.read_csv(directory + '/' +'V4_IMON_NOTES_DATA_TABLE.csv', encoding = 'latin1')
df3 = df3.loc[(df3['NOTE_TYPE'] == 'Progress Note, Outpatient') | (df3['NOTE_TYPE'] =='Progress Note, Inpatient')
             | (df3['NOTE_TYPE'] =='Consultation Note')| (df3['NOTE_TYPE'] =='History and Physical')]
df3 = df3.fillna('N/A')
df3 = df3[df3['NOTE']!='N/A']
df3['NOTE'] = df3['NOTE'].apply(encoding)
df3['NOTE'] = df3['NOTE'].astype('str')
#df3 = df3[df3['ANON_ID'].isin(ids)].reset_index(drop = True)
print ('Number of progress note' + str(df3.shape))
## radiology
df1 = pd.read_csv(directory + '/' + "Cleaned_Banerjee_onco_radio_reports.csv", encoding='latin1')
df1 = df1[df1['ANON_ID'].isin(ids)].reset_index(drop = True)
df1 = report_read(df1)
df1['NOTE_TYPE'] = 'RAD_REPORT'
print ('Number of radiology note' + str(df1.shape))
## pathology
df2 = pd.read_csv(directory + '/' + "Banerjee_onco_path_reports.csv.gz", sep = '|',compression='infer',encoding='latin1', index_col=False, error_bad_lines = False)
df2 = df2[df2['ANON_ID'].isin(ids)].reset_index(drop = True)
df2 = report_read(df2)
df2['NOTE_TYPE'] = 'PAT_REPORT'
print ('Number of pathology note' + str(df2.shape))



## merge notes

dfall = df1
dfall = dfall.append(df2, ignore_index = True, sort=True)
dfall = dfall.append(df3, ignore_index = True, sort=True)
print ('df all: ', dfall.shape)
del df1, df2, df3

dfall = dfall.fillna('N/A')
dfall['NOTE']=dfall['NOTE'].astype('str')
dfall = dfall[dfall.NOTE_DATE != 'N/A']
## preprocessing
print('Preprocessing data....')
notes = clean(dfall)
notes['NOTE_DATE'] =  pd.to_datetime(notes['NOTE_DATE'])

notes = notes[notes['NOTE_DATE']> '2008-03-01']
del dfall

## quarter-division
print('Quarter division....')
notes =  notes[notes['NOTE_DATE']!='N/A']
ANON_ID = notes['ANON_ID'].unique()
FIRST_ENCOUNTER = []
LAST_ENCOUNTER = []

for i in ANON_ID: 
    temp_df = notes[notes['ANON_ID']==i]
    temmp_df = temp_df.reset_index(drop=True)
    FIRST_ENCOUNTER.append(min(temp_df['NOTE_DATE']))
    LAST_ENCOUNTER.append(max(temp_df['NOTE_DATE']))

pat_df = pd.DataFrame({'ANON_ID':ANON_ID, 'FIRST_ENCOUNTER': FIRST_ENCOUNTER, 'LAST_ENCOUNTER':LAST_ENCOUNTER})
pat_df = pat_df[pat_df['LAST_ENCOUNTER']!='N/A']
pat_df['FIRST_ENCOUNTER'] =  pd.to_datetime(pat_df['FIRST_ENCOUNTER'])
pat_df['LAST_ENCOUNTER'] =  pd.to_datetime(pat_df['LAST_ENCOUNTER'])

pat_df.to_csv('./outcome/Patient_encounters.csv')

plus_month_period = 1
ID = []
Quarter = []
ANON_ID = pat_df['ANON_ID'].unique()
for i in ANON_ID: 
    temp_pat_df = pat_df[pat_df['ANON_ID']==i]
    C = temp_pat_df.iloc[0]['FIRST_ENCOUNTER']
    while (C+pd.DateOffset(months=plus_month_period)) < temp_pat_df.iloc[0].LAST_ENCOUNTER :
        ID.append(i)
        Quarter.append(C+ pd.DateOffset(months=plus_month_period));
        C = C+ pd.DateOffset(months=plus_month_period)
    ID.append(i)
    Quarter.append(temp_pat_df.iloc[0].LAST_ENCOUNTER);

pat_df = pd.DataFrame({'ANON_ID':ID, 'DATE': Quarter})
## Quarter-wise



ID = pat_df['ANON_ID'].unique()
SENT = []
TAG_SENT = []
UN_sent = []
RECURR = []
PAT = []
START_DATE =[]
END_DATE =[]
NOTE = []
f = 0
for ids in ID:
    temp_pat = pat_df[pat_df['ANON_ID']==ids]
    temp_pat = temp_pat.reset_index(drop = True)
    print(str(ids)+':'+str(temp_pat.shape[0]))
    for i in range(temp_pat.shape[0]-1):
        temp_df = notes[notes['ANON_ID']==ids]
        temp_df = temp_df.sort_values(by=['NOTE_DATE'])
        temp_df = temp_df.reset_index(drop=True)
        PAT.append(ids)
        SENT.append('<break>')
        TAG_SENT.append('<break>')
        UN_sent.append('<break>')
        NOTE.append('<break>')
        START_DATE.append(temp_pat.iloc[i]['DATE'])
        END_DATE.append(temp_pat.iloc[i+1]['DATE'])
        for j in range(temp_df.shape[0]):
            if temp_df.iloc[j]['NOTE_DATE']>= temp_pat.iloc[i]['DATE'] and temp_df.iloc[j]['NOTE_DATE'] <= temp_pat.iloc[i+1]['DATE']:
                SENT[f] = SENT[f] + '<break>' + str(temp_df.iloc[j]['SENTS'])
                TAG_SENT[f] = TAG_SENT[f] + '<break>' + str(temp_df.iloc[j]['Tagged_sent'])
                UN_sent[f] = UN_sent[f] + '<break>' + str(temp_df.iloc[j]['UNDER_SCORED_SENT'])
                NOTE[f] = NOTE[f] + '<NEXT_NOTE>' + str(temp_df.iloc[j]['NOTE'])
        f = f+1
        
        
qaurter_wise_prediction  = pd.DataFrame({'ANON_ID':PAT, 'START_DATE':START_DATE, 'END_DATE':END_DATE, 'SENT':SENT,'TAG_SENT':TAG_SENT, 'UN_SENT':UN_sent, 'NOTE':NOTE})
qaurter_wise_prediction['START_DATE'] =  pd.to_datetime(qaurter_wise_prediction['START_DATE'])
qaurter_wise_prediction['END_DATE'] =  pd.to_datetime(qaurter_wise_prediction['END_DATE'])
qaurter_wise_prediction.to_csv('./outcome/quarters.csv')

del notes
##classification

pred  = classification_test(qaurter_wise_prediction)
pred['START_DATE'] =  pd.to_datetime(pred['START_DATE'])
pred['END_DATE'] =  pd.to_datetime(pred['END_DATE'])
pred.to_excel('C:/Users/imonb/Box/Imon Banerjee\'s Files/Koo_project/Code/outcome/quarters_tag.xlsx')

pred  = classification_dense(qaurter_wise_prediction)
pred['START_DATE'] =  pd.to_datetime(pred['START_DATE'])
pred['END_DATE'] =  pd.to_datetime(pred['END_DATE'])
pred.to_excel('C:/Users/imonb/Box/Imon Banerjee\'s Files/Koo_project/Code/outcome/quarters_dense.xlsx')
