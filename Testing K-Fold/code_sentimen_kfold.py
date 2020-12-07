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

data = pd.read_csv('D:/SKRIPSI/Program Separated/Preprocessing Text/Hasil Preprocessing/preprocessing_dataset_sentimen_v3.tsv', header = None, delimiter='\t')
data.columns = ['Id','Nama','Text_Final','tokens', 'Label']

x=data['Text_Final']
y=data['Label']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

acc=[]
presisi=[]
recal=[]
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
fold = 1

for train, test in kfold.split(x, y):
    labelss= pd.DataFrame(columns=['Pos', 'Neg'])
    pos = []
    neg = []
    for l in y[train]:
      if l == 'Positif':
          pos.append(1)
          neg.append(0)
      elif l == 'Negatif':
          pos.append(0)
          neg.append(1)
    labelss['Pos']= pos
    labelss['Neg']= neg

    word2vec_path = 'C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_word2vec_v3.txt'
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 100

    x[train]=x[train].astype(str)

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

    label_names = ['Pos', 'Neg']

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
        preds = Dense(labels_index, activation='sigmoid')(x)
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

    hist = model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.2, shuffle=True , batch_size=batch_size,callbacks=[es_callback])
    predictions = model.predict(test_lstm_data, batch_size=1, verbose=1)

    labels = ['Positif','Negatif']
    prediction_labels=[]
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    y_test= y[test].values
    y_tes=[]
    for i in range(0, len(y_test)):
      y_tes.append(y_test[i])

    y_pred=[]
    for p in predictions:
        y_pred.append(labels[np.argmax(p)])

    cf_sentimen = pd.DataFrame(
    		    data=confusion_matrix(y_tes, y_pred, labels=labels),
    		    columns=labels,
    		    index=labels
    		)
    print(cf_sentimen)

    tps_sentimen = {}
    fps_sentimen  = {}
    fns_sentimen  = {}
    tns_sentimen  = {}
    for label in labels:
      tps_sentimen[label] = cf_sentimen.loc[label, label]
      fps_sentimen[label] = cf_sentimen[label].sum() - tps_sentimen[label]
      fns_sentimen[label] = cf_sentimen.loc[label].sum() - tps_sentimen[label]

    for label in set(y_tes):
      tns_sentimen[label] = len(y_tes) - (tps_sentimen[label] + fps_sentimen[label] + fns_sentimen[label])

    print(tps_sentimen)
    print(fps_sentimen)
    print(fns_sentimen)
    print(tns_sentimen)


    accuracySentimen=sum(tps_sentimen.values())/len(y_tes)
    acc.append(accuracySentimen)


    tpfp_sentimen = [ai + bi for ai, bi in zip(list(tps_sentimen.values()), list(fps_sentimen.values()))]
    precision=[ai / bi if bi>0 else 0 for ai , bi in zip(list(tps_sentimen.values()), tpfp_sentimen)]
    precisionSentimen=sum(precision)/2
    presisi.append(precisionSentimen)

    tpfn_sentimen = [ai + bi for ai, bi in zip(list(tps_sentimen.values()), list(fns_sentimen.values()))]
    recall=[ai / bi if bi>0 else 0 for ai, bi in zip(list(tps_sentimen.values()), tpfn_sentimen)]
    recallSentimen=sum(recall)/2
    recal.append(recallSentimen)

    y_tesss=[]
    for i in range(0, len(y_test)):
      if y_test[i]=='Positif':
        y_tesss.append(1)
      elif y_test[i]=='Negatif':
        y_tesss.append(0)

    labelsss=[1,0]

    y_predss=[]
    for p in predictions:
        y_predss.append(labelsss[np.argmax(p)])

    fpr, tpr, thresholds = roc_curve(y_tesss, y_predss)
    print("fpr:"+str(fpr))
    print("tpr:"+str(tpr))
    nilai_auc = auc(fpr, tpr)
    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(nilai_auc)
    ax.plot(mean_fpr, interp_tpr, label=r'ROC Fold %d (AUC = %f)' % (fold, nilai_auc),
        lw=2, alpha=0.3)
    fold=fold+1

hasil={'acc':acc,'presisi':presisi,'recall':recal}
print(hasil)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=4, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Kurva ROC Sentimen (5-Fold Cross Validation)")
ax.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
fig.savefig('D:/SKRIPSI/Program Separated/Testing K-Fold/Hasil Testing/ROC Sentimen.png')

# with open ('D:/SKRIPSI/Program Separated/Testing K-Fold/Hasil Testing/hasil_uji_sentimen_kfold.json','w') as json_file:
#   json.dump(hasil,json_file)