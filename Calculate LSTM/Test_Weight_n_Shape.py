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


data = pd.read_csv('D:/SKRIPSI/Dataset/LabelSentimen.tsv', header = None, delimiter='\t')

data.columns = ['Id','Nama','Text', 'Label']

data.head()
data.Label.unique()
data.shape

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
		    
data['Text_casefolding'] = data['Text'].apply(lambda x: case_folding(x))
print ("case folding")

def remove_punct(text):  
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', ' ', text)
    return text_nopunct

data['Text_NoPuct'] = data['Text_casefolding'].apply(lambda x: remove_punct(x))
print ("Remove punctuation")

def remove_num(text):  
    text_nonum = ''
    text_nonum = re.sub(r'\d+',' ', text)
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

data['Akhir'] = [remove_stop_words(sen) for sen in tokens] 
print ("stopword")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

result = [' '.join(sen) for sen in data['Akhir']]

data['Text_Final'] = result

data['tokens'] = data['Akhir']

data = data[['Text_Final', 'tokens', 'Label', 'Pos', 'Neg']]


data_train, data_test = train_test_split(data, test_size=0.01, random_state=42)

word2vec_path = 'C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/coba_word2vec.txt'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 5

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())

training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())
train_word_index = tokenizer.word_index
train_lstm_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))

for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_lstm_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

label_names = ['Pos', 'Neg']

y_train = data_train[label_names].values
x_train = train_lstm_data
y_tr = y_train

def recurrent_nn(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
		    
    embedding_layer = Embedding(num_words,
		                            embedding_dim,
		                            weights=[embeddings],
		                            input_length=max_sequence_length,
		                            trainable=False)
  
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm = LSTM(5)(embedded_sequences)
    preds = Dense(labels_index, activation='sigmoid')(lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
		                  optimizer='adam',
		                  metrics=['acc'])
    model.summary()
    return model

model = recurrent_nn(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
		                len(list(label_names)))


print(model.layers[2].trainable_weights)

W = model.layers[2].get_weights()[0]
U = model.layers[2].get_weights()[1]
b = model.layers[2].get_weights()[2]
units = int(int(model.layers[2].trainable_weights[0].shape[1])/4)

W_i = W[:, :units]
W_f = W[:, units: units * 2]
W_c = W[:, units * 2: units * 3]
W_o = W[:, units * 3:]

U_i = U[:, :units]
U_f = U[:, units: units * 2]
U_c = U[:, units * 2: units * 3]
U_o = U[:, units * 3:]

b_i = b[:units]
b_f = b[units: units * 2]
b_c = b[units * 2: units * 3]
b_o = b[units * 3:]

print("Wi")
print(W_i)
print("Ui")
print(U_i)
print("bi")
print(b_i)



#Testing
dats=["pantai indrayanti indah bersih"]
training_sequences2= tokenizer.texts_to_sequences(dats)
xx = pad_sequences(training_sequences2, maxlen=MAX_SEQUENCE_LENGTH)
output=model.predict(xx)
print(output)
