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
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

data = pd.read_csv('D:/SKRIPSI/Program Separated/Preprocessing Text/Hasil Preprocessing/preprocessing_dataset_sentimen_faizal.tsv', header = None, delimiter='\t')

data.columns = ['Id','Text_Final', 'Kategori']

x=data['Text_Final']
y=data['Kategori']

kfold = StratifiedKFold(n_splits=2, shuffle=False, random_state=0)

for train, test in kfold.split(x, y): 
    labelss= pd.DataFrame(columns=['neg','pos','net'])
    neg = []
    pos = []
    net = []
    for l in y[train]:
        if l == 0:
            neg.append(1)
            pos.append(0)
            net.append(0)
        if l == 1:
            neg.append(0)
            pos.append(1)
            net.append(0)
        elif l == 2:
            neg.append(0)
            pos.append(0)
            net.append(1)

    labelss['neg']= neg
    labelss['pos']= pos
    labelss['net']= net

    word2vec_path = 'D:/SKRIPSI/word_v1.txt'
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    MAX_SEQUENCE_LENGTH = 60
    EMBEDDING_DIM = 300

    x[train]=x[train].astype(str)
    x[test]=x[test].astype(str)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x[train].tolist())
    train_word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(x[train].tolist())
    train_lstm_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    test_sequences = tokenizer.texts_to_sequences(x[test].tolist())
    test_lstm_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
    for word,index in train_word_index.items():
        train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

    y_train = labelss.values
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
        preds = Dense(labels_index, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['acc'])
        model.summary()
        return model


    model = recurrent_nn(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                        len(list(labelss)))

    num_epochs = 50
    batch_size = 32

    es_callback =EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    hist = model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.2, shuffle=True , batch_size=batch_size,callbacks=[es_callback])

    predictions = model.predict(test_lstm_data, batch_size=1, verbose=1)

    