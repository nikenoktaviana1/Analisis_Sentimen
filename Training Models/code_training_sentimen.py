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

data = pd.read_csv('D:/SKRIPSI/Dataset/LabelSentimen_v3.tsv', header = None, delimiter='\t')
data.columns = ['Id','Nama','Text', 'Label']

pos = []
neg = []
for l in data.Label:
    if l == 'Positif':
        pos.append(1)
        neg.append(0)
    elif l == 'Negatif':
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

data['Text_casefolding'] = data['Text'].apply(lambda x: case_folding(x))
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

data = data[['Text_Final', 'tokens', 'Label', 'Pos', 'Neg']]

data_train, data_test = train_test_split(data, test_size=0.00001, random_state=42)

model = Word2Vec(data['tokens']  , size=100, min_count=1)
model.wv.save_word2vec_format('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_word2vec_v3.txt',binary=False)

word2vec_path = 'C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_word2vec_v3.txt'
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

labels = ['Positif','Negatif']

prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

sum(data_test.Label==prediction_labels)/len(prediction_labels)


model_json = model.to_json()
with open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_sentimen_v3.json','w') as json_file:
  json_file.write(model_json)
model.save_weights("D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_sentimen_v3.h5")
print("Model saved to disk")

with open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/Saved_Tokenize_v3.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.plot(hist.history['loss'],'r',linewidth=1.0)
plt.plot(hist.history['val_loss'],'b',linewidth=1.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=10)
plt.xlabel('Epochs ',fontsize=10)
plt.ylabel('Loss',fontsize=10)
plt.title('Loss Model Sentimen',fontsize=13)
fig1.savefig('D:/SKRIPSI/Program Separated/Training Models/Plot_training/loss_sentimen.png')
plt.show()

fig2=plt.figure()
plt.plot(hist.history['acc'],'r',linewidth=1.0)
plt.plot(hist.history['val_acc'],'b',linewidth=1.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10)
plt.xlabel('Epochs ',fontsize=10)
plt.ylabel('Accuracy',fontsize=10)
plt.title('Accuracy Model Sentimen',fontsize=13)
fig2.savefig('D:/SKRIPSI/Program Separated/Training Models/Plot_training/acc_sentimen.png')
plt.show()

