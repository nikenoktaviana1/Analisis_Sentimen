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

data = pd.read_csv('D:/SKRIPSI/Program Separated/Preprocessing Text/Hasil Preprocessing/preprocessing_dataset_kategori.tsv', header = None, delimiter='\t')

data.columns = ['Id','Nama','Text_Final','tokens', 'Kategori']

x=data['Text_Final']
y=data['Kategori']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

acc=[]
presisi=[]
recal=[]
fold = 0
kelas0_tpr = []
kelas0_auc = []
kelas1_tpr = []
kelas1_auc = []
kelas2_tpr = []
kelas2_auc = []
kelas3_tpr = []
kelas3_auc = []
micro_tpr=[]
micro_auc=[]
all_fpr0=[]
all_tpr0=[]
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for train, test in kfold.split(x, y): 
    labelss= pd.DataFrame(columns=['dayatarik','aksesbilitas','kebersihan','fasilitas'])
    dayatarik = []
    aksesbilitas = []
    kebersihan = []
    fasilitas = []
    for l in y[train]:
        if l == 'Daya Tarik':
            dayatarik.append(1)
            aksesbilitas.append(0)
            kebersihan.append(0)
            fasilitas.append(0)
        if l == 'Aksesbilitas':
            dayatarik.append(0)
            aksesbilitas.append(1)
            kebersihan.append(0)
            fasilitas.append(0)
        if l == 'Kebersihan':
            dayatarik.append(0)
            aksesbilitas.append(0)
            kebersihan.append(1)
            fasilitas.append(0)
        elif l == 'Fasilitas':
            dayatarik.append(0)
            aksesbilitas.append(0)
            kebersihan.append(0)
            fasilitas.append(1)

    labelss['dayatarik']= dayatarik
    labelss['aksesbilitas']= aksesbilitas
    labelss['kebersihan']= kebersihan
    labelss['fasilitas']= fasilitas

    word2vec_path = 'C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_word2vec.txt'
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIM = 100

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

    labels = ['Daya Tarik','Aksesbilitas','Kebersihan','Fasilitas']

    prediction_labels=[]
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])

    xx=sum(y[test]==prediction_labels)/len(prediction_labels)

    y_test= y[test].values

    y_tes=[]
    for i in range(0, len(y_test)):
      y_tes.append(y_test[i])

    y_pred=[]
    for p in predictions:
        y_pred.append(labels[np.argmax(p)])

    cf_kategori = pd.DataFrame(
    data=confusion_matrix(y_tes, y_pred, labels=labels),
    columns=labels,
    index=labels
    )
    print(cf_kategori)

    tps_kategori = {}
    fps_kategori  = {}
    fns_kategori  = {}
    tns_kategori  = {}
    for label in labels:
      tps_kategori[label] = cf_kategori.loc[label, label]
      fps_kategori[label] = cf_kategori[label].sum() - tps_kategori[label]
      fns_kategori[label] = cf_kategori.loc[label].sum() - tps_kategori[label]
          
    for label in set(y_tes):
      tns_kategori[label] = len(y_tes) - (tps_kategori[label] + fps_kategori[label] + fns_kategori[label])

    print(tps_kategori)
    print(fps_kategori)
    print(fns_kategori)
    print(tns_kategori)
    accuracyKategori=sum(tps_kategori.values())/len(y_tes)
    acc.append(accuracyKategori)


    tpfp_kategori = [ai + bi for ai, bi in zip(list(tps_kategori.values()), list(fps_kategori.values()))]
    precision=[ai / bi  if bi>0 else 0 for ai, bi in zip(list(tps_kategori.values()), tpfp_kategori)]
    precisionKategori=sum(precision)/4
    presisi.append(precisionKategori)

    tpfn_kategori = [ai + bi for ai, bi in zip(list(tps_kategori.values()), list(fns_kategori.values()))]
    recall=[ai / bi  if bi>0 else 0 for ai, bi in zip(list(tps_kategori.values()), tpfn_kategori)]
    recallKategori=sum(recall)/4
    recal.append(recallKategori)


    labelsss=[1,2,3,4]
    y_predss=[]
    for p in predictions:
        y_predss.append(labelsss[np.argmax(p)])

    y_binari=label_binarize(y[test],classes=['Daya Tarik','Aksesbilitas','Kebersihan','Fasilitas'])
    n_classes=y_binari.shape[1]

    y_prediksi=label_binarize(y_predss,classes=[1,2,3,4])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
      fpr[i],tpr[i], _ = roc_curve(y_binari[:,i], y_prediksi[:,i])
      roc_auc[i]=auc(fpr[i],tpr[i])

    print("fpr=")
    print(fpr)
    print("tpr=")
    print(tpr)


    fpr["micro"],tpr["micro"], _ = roc_curve(y_binari.ravel(), y_prediksi.ravel())
    roc_auc["micro"]=auc(fpr["micro"],tpr["micro"])

    interp_tpr_micro = interp(mean_fpr, fpr["micro"], tpr["micro"])
    interp_tpr_micro[0] = 0.0
    micro_tpr.append(interp_tpr_micro)
    micro_auc.append(roc_auc["micro"])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr/=n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"],tpr["macro"])
    colors=cycle(['blue','orange','black','red'])

    for i, color in zip(range(n_classes),colors):
      if i == 0 :
        interp_tpr = interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        kelas0_tpr.append(interp_tpr)
        kelas0_auc.append(roc_auc[i])
      elif i == 1 :
        interp_tpr = interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        kelas1_tpr.append(interp_tpr)
        kelas1_auc.append(roc_auc[i])
      elif i == 2 :
        interp_tpr = interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        kelas2_tpr.append(interp_tpr)
        kelas2_auc.append(roc_auc[i])
      elif i == 3 :
        interp_tpr = interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        kelas3_tpr.append(interp_tpr)
        kelas3_auc.append(roc_auc[i])
      ax.plot(fpr[i],tpr[i], color=color, lw=1)

    plt.plot([0,1],[0,1],'k--', lw=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Kategori K=' +str(fold))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig('D:/SKRIPSI/Program Separated/Testing K-Fold/Hasil Testing/ROC Kategori K=' +str(fold)+'.png')
    fold=fold+1

mean_tpr0 = np.mean(kelas0_tpr, axis=0)
mean_tpr0[-1] = 1.0
mean_auc0 = auc(mean_fpr, mean_tpr0)

mean_tpr1 = np.mean(kelas1_tpr, axis=0)
mean_tpr1[-1] = 1.0
mean_auc1 = auc(mean_fpr, mean_tpr1)

mean_tpr2 = np.mean(kelas2_tpr, axis=0)
mean_tpr2[-1] = 1.0
mean_auc2 = auc(mean_fpr, mean_tpr2)

mean_tpr3 = np.mean(kelas3_tpr, axis=0)
mean_tpr3[-1] = 1.0
mean_auc3 = auc(mean_fpr, mean_tpr3)

mean_micro=np.mean(micro_tpr,axis=0)
mean_micro[-1] = 1.0
mean_auc_micro = auc(mean_fpr, mean_micro)


ax.plot(mean_fpr, mean_micro,lw=3, label='micro (area = {0:0.2f})'''.format(mean_auc_micro),color='deeppink',linestyle=':')

mean_tpr = (mean_tpr0 + mean_tpr1 + mean_tpr2 + mean_tpr3)/4
mean_auc = (mean_auc0 + mean_auc1 + mean_auc2 + mean_auc3)/4
ax.plot(mean_fpr, mean_tpr, alpha=.8,label='macro (area = {0:0.2f})'''.format(mean_auc),color='green',linestyle=':',linewidth=3)
ax.plot([0, 1], [0, 1], linestyle='--', lw=3, color='pink',
        label='Chance')
ax.plot(mean_fpr, mean_tpr0,
        label=r'Mean ROC Kelas 0 (AUC = %0.2f)' % (mean_auc0),
        lw=4,color='blue')
ax.plot(mean_fpr, mean_tpr1,
        label=r'Mean ROC Kelas 1 (AUC = %0.2f)' % (mean_auc1),
        lw=4, color='orange')
ax.plot(mean_fpr, mean_tpr2,
        label=r'Mean ROC Kelas 2 (AUC = %0.2f)' % (mean_auc2),
        lw=4, color='black')
ax.plot(mean_fpr, mean_tpr3,
        label=r'Mean ROC Kelas 3 (AUC = %0.2f)' % (mean_auc3),
        lw=4,color='red')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Kurva ROC Kategori")
ax.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
fig.savefig('D:/SKRIPSI/Program Separated/Testing K-Fold/Hasil Testing/Kurva ROC Kategori.png')

hasil={'acc':acc,'presisi':presisi,'recall':recal}
print(hasil)
with open ('D:/SKRIPSI/Program Separated/Testing K-Fold/Hasil Testing/hasil_uji_kategori_kfold.json','w') as json_file:
  json.dump(hasil,json_file)
