from keras.models import load_model
from keras.models import model_from_json

import string
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
import sys
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from nltk import word_tokenize, WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pymysql

json_file = open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_kategori_v3.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_kategori_v3.h5')

json_file2 = open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_sentimen_v3.json', 'r')
model_json2 = json_file2.read()
json_file2.close()
model_kategori = model_from_json(model_json2)
model_kategori.load_weights('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_sentimen_v3.h5')

with open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_tokenize_v3.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


data = pd.read_csv('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/DATA/FILE TSV/data_pantai_mentah.tsv', header = None, delimiter='\t')

data.columns = ['Id','Nama Pantai','Text']

def case_folding(tokens): 
    return tokens.lower()  
        
data['Text_casefolding'] = data['Text'].apply(lambda x: case_folding(x))
print ("case folding")


def remove_punct(text):  
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

data['Text_NoPuct'] = data['Text_casefolding'].apply(lambda x: remove_punct(x))
print ("Remove punctuation")


def remove_num(text):  
    text_nonum = ''
    text_nonum = re.sub(r'\d+','', text)
    return text_nonum

data['Text_Nonum'] = data['Text_NoPuct'].apply(lambda x: remove_num(x))
print ("Remove Number")

def open_kamus_prepro(x):
  kamus={}
  with open(x,'r') as file :
    for line in file :
      slang=line.replace("'","").split(':')
      kamus[slang[0].strip()]=slang[1].rstrip('\n').lstrip()
  return kamus

kamus_slang=open_kamus_prepro('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Kamus Preprocessing/Kamus slang_word.txt')

def slangword(text):  
    sentence_list = text.split()
    new_sentence = []
    for word in sentence_list:
      for candidate_replacement in kamus_slang:
        if candidate_replacement == word:
          word = word.replace(candidate_replacement, kamus_slang[candidate_replacement])
      new_sentence.append(word)
    return " ".join(new_sentence)

data['Text_slang'] = data['Text_Nonum'].apply(lambda x: slangword(x))
print ("slangword")

kamus_negasi=open_kamus_prepro('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Kamus Preprocessing/Kamus negation_word.txt')

def ganti_negasi(w):
  w_splited = w.split(' ')
  if 'tidak' in w_splited:
     index_negasi = w_splited.index('tidak')
     for i,k in enumerate(w_splited):
       if k in kamus_negasi and w_splited[i-1] == 'tidak':
         w_splited[i] = kamus_negasi[k]

  return ' '.join(w_splited)

data['negasi'] = data['Text_slang'].apply(lambda x: ganti_negasi(x))
print ("negasi")

tokens = [word_tokenize(sen) for sen in data.negasi]

kamus_stopword=[]
with open('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Kamus Preprocessing/Kamus stopword.txt','r') as file :
  for line in file :
    slang=line.replace("'","").strip()
    kamus_stopword.append(slang)

def remove_stop_words(tokens):
    return [word for word in tokens if word not in kamus_stopword]

data['filtered_words'] = [remove_stop_words(sen) for sen in tokens] 
print ("stopword")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(tokens):  
    data_stem =[]
    for i in tokens:
      kata = stemmer.stem(i)
      data_stem.append(kata)
    return data_stem

data['Akhir'] = data['filtered_words'].apply(lambda x: stemming(x))
print ("stemming")

result = [' '.join(sen) for sen in data['Akhir']]

data['Text_Final'] = result

data['tokens'] = data['Akhir']

data = data[['Id','Nama Pantai','Text','Text_Final', 'tokens']]

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100

sequences = tokenizer.texts_to_sequences(data['tokens'])

word_index = tokenizer.word_index
print('Number of Unique Tokens',len(word_index))


test_cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = test_cnn_data

print(x_test)

prediksi = model.predict(x_test, batch_size=1, verbose=1)

prediksi_kategori = model_kategori.predict(x_test, batch_size=1, verbose=1)

class_category = ['Daya Tarik', 'Aksesbilitas','Kebersihan','Fasilitas']

class_sentimen = ['Positif', 'Negatif']

for i in range(prediksi_kategori.shape[0]):
  print(str(i) + " " + class_sentimen[prediksi_kategori[i].argmax()] + " , " + class_category[prediksi[i].argmax()] + " : " + data['Text_Final'][i])

conn = pymysql.connect(
    host='localhost',
    user='root',
    passwd='',
    port=3306,
    db='database_analisis_sentimen'
)

def get_list_predicted(matrix_res,data,prediksi):
  res_data = []
  for i in range(matrix_res.shape[0]):
    pred_data = {
        'sentimen' : class_sentimen[matrix_res[i].argmax()],
        'kategori' : class_category[prediksi[i].argmax()],
        'text_final' : data['Text_Final'][i],
        'id_pantai' : data['Id'][i],
        'nama_pantai': data['Nama Pantai'][i]

    }
    res_data.append(pred_data)
  return res_data
  
def insert_database(conn,data):
  with conn.cursor() as cursor:
    for d in data:
      sql = "insert into hasil_klasifikasi_2 values(0,'{}','{}','{}','{}')".format(d['id_pantai'],d['text_final'],d['sentimen'],d['kategori'])
      print("inserting %s" %sql)
      cursor.execute(sql)
  conn.commit()

dict_data  = get_list_predicted(prediksi_kategori,data,prediksi)
insert_database(conn,dict_data)

