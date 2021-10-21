import pandas as pd
import numpy as np
import csv
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import regex as re  # This is important otherwise re.py gives error
from nltk import sent_tokenize
import datetime
date_format = '%Y-%m-%d'
#nltk.download('punkt')
low_memory=False


wordMap = {}
terms = []
with open('./dic/pqrs_added_ak.txt', 'r') as termFile:
    for line in termFile:
        line = line.encode().decode('utf-8')
        words = line.split('\t')
        if len(words) == 3:
            wordMap[' ' + words[1].lstrip(' ').rstrip(' ') + ' '] = ' ' + words[2].replace('\n', '').lstrip(' ').rstrip(' ') + ' '
            terms.append(' ' + words[1].lstrip(' ').rstrip(' ') + ' ')
terms = sorted(terms, key=len, reverse=True)


#compute the word map based on common-term dictionary
GeneralwordMap = {}
Gterms = []
GDterms = []
with open('./dic/clever_base_terminologyv2.txt', 'r') as termFile:
    for line in termFile:
        words = line.split('|')
        if len(words) == 3:
            GeneralwordMap[' ' + words[1].lstrip(' ').rstrip(' ') + ' '] = ' ' + words[2].replace('\r', '').replace('\n','').lstrip(' ').rstrip(' ') + ' '
            Gterms.append(' ' + words[1].lstrip(' ').rstrip(' ') + ' ')
            GDterms.append(words[2].replace('\r', '').replace('\n','').lstrip(' ').rstrip(' '))
Gterms = sorted(Gterms, key=len, reverse=True)


def correct_dates(notedate):
    if notedate !='N/A':
        try:
            t = str(pd.to_datetime(notedate).isoformat())
            notedate = datetime.datetime.strptime(t.split('T')[0], date_format)
        except:
            print (notedate)
    return notedate

def encoding(note):
    updated_note = note.encode("ascii", "ignore")
    return updated_note
header = r"\w+\s?:"
#header = r"\b[A-Z]*[a-z]*?\s?:"
headers = re.compile(header)

sent_end = r"\w+\."
sent_ends = re.compile(sent_end)

def replace_header(match):
    text = match.group()
    text = ' '.join(text.lstrip().rstrip().split())
    text = ' . ' + text 
    return text

def replace_sent_end(match):
    text = match.group()
    text = ' '.join(text.lstrip().rstrip().split())
    text = text.replace('.', ' ') + ' . '
    return text
s = r"\w+/\w+/(\w+)?/?(\w+)?"
signature = re.compile(s)
def replace_signature(text):
    text = re.sub(signature, ' signature:', text)
    text2 = text.replace('i have personally reviewed', ' signature:')
    text2 = text2.replace('cytotechnologist electronically signed', ' signature:')
    text3 = text2.replace('electronically signed', ' signature:')
    text3 = text3.replace('cytotechnologist', ' signature:')
    return text3

def get_rid_of_signature(note):
    clean_note = ''
    if 'signature:'in note:
        clean_note = note.split('signature:')[0]
    else:
        clean_note = note
    return clean_note

def clean_names(text):
    cleaned_note = ''
    cleaned_note = text.replace('ct(ascp)', ' ')
    cleaned_note = cleaned_note.replace('troxell/nowels', ' ')
    cleaned_note = cleaned_note.replace('haas/mckenney', ' ')
    cleaned_note = cleaned_note.replace('haas/berry', ' ')
    cleaned_note = cleaned_note.replace('rama arumilli', ' ')
    cleaned_note = cleaned_note.replace('kunder/moore/schwartz', ' ')
    cleaned_note = cleaned_note.replace('tala lo-guyamata', ' ')
    cleaned_note = cleaned_note.replace('longacre/cesca', ' ')
    cleaned_note = cleaned_note.replace('lewis/kong', ' ')
    cleaned_note = cleaned_note.replace('wu/lau/isaza/e', ' ')
    cleaned_note = cleaned_note.replace('dimaio/long/schwartz', ' ')
    cleaned_note = cleaned_note.replace('xingnu luo', ' ')
    cleaned_note = cleaned_note.replace('anderson/vanderjagt/berry/hendrickson/kong', ' ')
    cleaned_note = cleaned_note.replace('burtelow/bindu/nowels', ' ')
    cleaned_note = cleaned_note.replace('karen e king c', ' ')
    cleaned_note = cleaned_note.replace('see comment', ' ')
    cleaned_note = cleaned_note.replace('runge/kambham', ' ')
    cleaned_note = cleaned_note.replace('howell/kong', ' ')
    cleaned_note = cleaned_note.replace('haynes/kong', ' ')
    cleaned_note = cleaned_note.replace('van de rijn', ' ')
    cleaned_note = cleaned_note.replace('manning/jensen', ' ')
    cleaned_note = cleaned_note.replace('treynor/rouse', ' ')
    cleaned_note = cleaned_note.replace('cherry/ ', ' ')
    cleaned_note = cleaned_note.replace('(see comment)', ' ')
    cleaned_note = cleaned_note.replace('torres-', ' ')
    cleaned_note = cleaned_note.replace('longacre/', ' ')
    cleaned_note = cleaned_note.replace('runge/', ' ')
    cleaned_note = cleaned_note.replace('treynor/longacre', ' ')
    cleaned_note = cleaned_note.replace('fan/longacre', ' ')
    cleaned_note = cleaned_note.replace('cherry/berry', ' ')
    cleaned_note = cleaned_note.replace('west/cherry', ' ')
    cleaned_note = cleaned_note.replace(' rouse ', ' ')
    cleaned_note = cleaned_note.replace(' west ', ' ')
    cleaned_note = cleaned_note.replace('summary table', ' ')
    cleaned_note = cleaned_note.replace('ajcc, 7th edition', ' ')
    return cleaned_note

def pre_process(note):
    note = re.sub(headers, replace_header, note)
    note = re.sub(sent_ends, replace_sent_end, note)
    note = note.replace("\r", ' . ')
    #note = note.replace("\s{3,}", ' \n ')
    note = note.replace("--", ' . ')
    #note = note.replace("\t", ' . ')
    note = note.replace("_", ' ')
    note = note.lower()
    note = note.replace('see reference below', ' ')
    note = note.replace('see comment', ' ')
    note = re.sub(r'[^\x00-\x7f]', r' ', note)
    for term in Gterms:
            if term in note:
                note = note.replace(term, GeneralwordMap[term])
    text = " ".join(note.split( ))
    return text

def clean(all_notes):
    output = []
    note_id = []
    all_sent = []
    original_note = []
    corrected_date = []
    note_type = []
    underscored_sent = []
    tagged_sent_final = []

    for index, row in all_notes.iterrows():
        note_id.append(row['ANON_ID'])
        note_type.append(row['NOTE_TYPE'])
        date = row['NOTE_DATE']
        corrected_date.append(correct_dates(date))
        note = row['NOTE']
        #original_note = u' '.join(note).encode('utf-8').strip()
        #original_note = str(original_note)
        note = pre_process(note)
        original_note.append(note)
        #sentences =  note.split('<break>')
        final_sent = ''
        all_sent_final = ''
        tagged_sent = ''
        underscored_sent1, underscored_sent2 = '', '' 
        sentences = sent_tokenize(note)
        for sent in sentences:
        #if any(word in sent for word in terms):
            for term in terms:
                if term in sent and sent not in all_sent_final:
                    final_sent = replace_signature(sent)
                    final_sent = get_rid_of_signature(final_sent)
                    final_sent = clean_names(" ".join(final_sent.split(' ')))
                    all_sent_final = all_sent_final + " <break> " + final_sent + " <break> "
                    if len(term.split()) > 1:
                        underscored_sent1 = final_sent.replace(term.strip(' ').rstrip(' '), re.sub(r"\s", '_', term.strip(' ').rstrip(' ')))
                        underscored_sent2 = underscored_sent2 +  " <break> " + underscored_sent1 + " <break> " 
    
        all_sent_final = " ".join(all_sent_final.split())
        underscored_sent2 = " ".join(underscored_sent2.split())
        tagged_sent = all_sent_final
        for t in terms:
            if t in all_sent_final:
                tagged_sent = tagged_sent.replace(t, wordMap[t])
        all_sent.append(all_sent_final)
        underscored_sent.append(underscored_sent2)
        tagged_sent_final.append(tagged_sent)
        print(all_sent_final)

    output = pd.DataFrame({'REPORT_ID':index,'ANON_ID': note_id, 'SENTS': all_sent, 'NOTE': original_note, 'NOTE_DATE':corrected_date, 'NOTE_TYPE':note_type, 'UNDER_SCORED_SENT':underscored_sent, 'Tagged_sent':tagged_sent_final})
    output = output.fillna('N/A')
    output.to_csv('./outcome/preprocessed_notes.csv')
    return output
    
