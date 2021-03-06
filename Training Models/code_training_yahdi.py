from __future__ import division, print_function
import nltk
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import json
import pickle

data = pd.read_csv('D:/SKRIPSI/Dataset/test3.csv', encoding='ISO-8859-1')

pos = []
neg = []
for l in data.Kelas:
    if l == 'Fakta':
        pos.append(1)
        neg.append(0)
    elif l == 'Hoax':
        pos.append(0)
        neg.append(1)

data['Pos']= pos
data['Neg']= neg
data.head()

def case_folding(tokens): 
    return tokens.lower()  

def remove_punct(text):  
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', ' ', text)
    return text_nopunct

def remove_num(text):  
    text_nonum = ''
    text_nonum = re.sub(r'\d+',' ', text)
    return text_nonum

def open_kamus_prepro(x):
  kamus={}
  with open(x,'r') as file :
    for line in file :
      slang=line.replace("'","").split(':')
      kamus[slang[0].strip()]=slang[1].rstrip('\n').lstrip()
  return kamus

kamus_slang=open_kamus_prepro('D:/SKRIPSI/Program Separated/Preprocessing Text/Kamus Preprocessing/Kamus spelling_word.txt')
def slangword(text):  
    sentence_list = text.split()
    new_sentence = []
    for word in sentence_list:
      for candidate_replacement in kamus_slang:
        if candidate_replacement == word:
          word = word.replace(candidate_replacement, kamus_slang[candidate_replacement])
      new_sentence.append(word)
    return " ".join(new_sentence)

kamus_negasi=open_kamus_prepro('D:/SKRIPSI/Program Separated/Preprocessing Text/Kamus Preprocessing/Kamus negation_word.txt')
def ganti_negasi(w):
  w_splited = w.split(' ')
  if 'tidak' in w_splited:
     index_negasi = w_splited.index('tidak')
     for i,k in enumerate(w_splited):
       if k in kamus_negasi and w_splited[i-1] == 'tidak':
         w_splited[i] = kamus_negasi[k]

  return ' '.join(w_splited)

kamus_stopword=[]
with open('D:/SKRIPSI/Program Separated/Preprocessing Text/Kamus Preprocessing/Kamus stopword.txt','r') as file :
  for line in file :
    slang=line.replace("'","").strip()
    kamus_stopword.append(slang)

def remove_stop_words(tokens):
    return [word for word in tokens if word not in kamus_stopword]

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(tokens):  
    data_stem =[]
    for i in tokens:
      kata = stemmer.stem(i)
      data_stem.append(kata)
    return data_stem

data['Text_casefolding'] = data['Konten'].apply(lambda x: case_folding(x))
print ("case folding")

data['Text_NoPuct'] = data['Text_casefolding'].apply(lambda x: remove_punct(x))
print ("Remove punctuation")

data['Text_Nonum'] = data['Text_NoPuct'].apply(lambda x: remove_num(x))
print ("Remove Number")

data['Text_slang'] = data['Text_Nonum'].apply(lambda x: slangword(x))
print ("slangword")

data['negasi'] = data['Text_slang'].apply(lambda x: ganti_negasi(x))
print ("negasi")

tokens = [word_tokenize(sen) for sen in data.negasi]

data['filtered_words'] = [remove_stop_words(sen) for sen in tokens] 
print ("stopword")

data['Akhir'] = data['filtered_words'].apply(lambda x: stemming(x))
print ("stemming")

result = [' '.join(sen) for sen in data['Akhir']]
data['Text_Final'] = result
data['tokens'] = data['Akhir']

data = data[['Text_Final', 'tokens', 'Kelas', 'Pos', 'Neg']]

data_train, data_test = train_test_split(data, test_size=0.00001, random_state=42)

model = Word2Vec(data['tokens']  , size=100, min_count=1)
model.wv.save_word2vec_format('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/yahdi_word2vec.txt',binary=False)

word2vec_path = 'C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/yahdi_word2vec.txt'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
train_word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())
train_lstm_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_lstm_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

label_names = ['Pos', 'Neg']

y_train = data_train[label_names].values
x_train = train_lstm_data

def recurrent_nn(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
		    
    embedding_layer = Embedding(num_words,
		                            embedding_dim,
		                            weights=[embeddings],
		                            input_length=max_sequence_length,
		                            trainable=True)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm = LSTM(256)(embedded_sequences)
    x = Dense(128, activation='relu')(lstm)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
		                  optimizer='adam',
		                  metrics=['acc'])
    model.summary()
    return model

model = recurrent_nn(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
		                len(list(label_names)))

num_epochs = 100
batch_size = 32

es_callback =EarlyStopping(monitor='val_loss', patience=10, verbose=1)

hist = model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1, shuffle=True , batch_size=batch_size,callbacks=[es_callback])
predictions = model.predict(test_lstm_data, batch_size=1, verbose=1)

labels = ['Fakta','Hoax']

prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

sum(data_test.Kelas==prediction_labels)/len(prediction_labels)

# text='karbon dioksida'


# data_casefolding=[]
# for i in range(0, len(text)):
#   data_casefolding.append(case_folding(text[i]))
  


# data_remove=[]
# for i in range(0, len(text)):
#  data_remove.append(remove_punct(data_casefolding[i]))



# data_remove_num=[]
# for i in range(0, len(text)):
#  data_remove_num.append(remove_num(data_remove[i]))




# slang=[]
# for i in range(0, len(text)):
#  slang.append(slangword(data_remove_num[i]))



# negasi=[]
# for i in range(0, len(text)):
#   negasi.append(ganti_negasi(slang[i]))


# tokens = [word_tokenize(sen) for sen in negasi]





# filtered_words = [remove_stop_words(sen) for sen in tokens]

# stem=[]
# for i in range(0, len(text)):
#  stem.append(stemming(filtered_words[i]))

# print(stem)
# MAX_SEQUENCE_LENGTH = 50
# EMBEDDING_DIM = 300   


# prepro = [' '.join(sen) for sen in stem]
# sequences = tokenizer.texts_to_sequences(stem)

# word_index = tokenizer.word_index
# print('Number of Unique Tokens',len(word_index))

# test_cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# x_test = test_cnn_data

# print(x_test)

# prediksi_kategori = model_kategori.predict(x_test, batch_size=1, verbose=1)