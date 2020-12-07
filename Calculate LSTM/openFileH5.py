import h5py
import numpy as np

with h5py.File('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_model_kategori.h5','r') as hdf:
  ls= list(hdf.keys())
  print("list : "+ str(ls))
  data = hdf['lstm_1']
  # lstm1=np.array(data)

  print(data)

# from keras.models import load_model
# from keras.models import model_from_json
# import string
# import numpy as np
# import pandas as pd
# import pickle
# from collections import defaultdict
# import re
# import sys
# import os
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Embedding
# from keras.layers import Dense, Input, Flatten
# from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
# from keras.models import Model

# with open('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_tokenize.pickle', 'rb') as handle:
# 	tokenizer = pickle.load(handle)

# json_file_kategori = open('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_model_kategori.json', 'r')
# model_json_kategori = json_file_kategori.read()
# json_file_kategori.close()
# model_kategori = model_from_json(model_json_kategori)
# model_kategori.load_weights('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_model_kategori.h5')

# model_kategori.summary()
# # print(model_kategori.layers[1].trainable_weights)
# # print(model_kategori.layers[2].trainable_weights)

# W = model_kategori.layers[2].get_weights()[0]
# U = model_kategori.layers[2].get_weights()[1]
# b = model_kategori.layers[2].get_weights()[2]
# units = int(int(model_kategori.layers[2].trainable_weights[0].shape[1])/4)

# W_i = W[:, :units]
# W_f = W[:, units: units * 2]
# W_c = W[:, units * 2: units * 3]
# W_o = W[:, units * 3:]

# U_i = U[:, :units]
# U_f = U[:, units: units * 2]
# U_c = U[:, units * 2: units * 3]
# U_o = U[:, units * 3:]

# b_i = b[:units]
# b_f = b[units: units * 2]
# b_c = b[units * 2: units * 3]
# b_o = b[units * 3:]

# print("Wi")
# print(W_i)
# print("Ui")
# print(U_i)
# print("bi")
# print(b_i)
# print(len(U_i))



# #Testing
# dats=["pantai indrayanti indah bersih"]
# training_sequences2= tokenizer.texts_to_sequences(dats)
# xx = pad_sequences(training_sequences2, maxlen=50)
# output=model_kategori.predict(xx)
# print(output)