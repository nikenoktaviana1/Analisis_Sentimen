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


# json_file_kategori = open('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Model Saved/saved_model_kategori.json', 'r')
json_file_kategori = open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_kategori_v3.json', 'r')
model_json_kategori = json_file_kategori.read()
json_file_kategori.close()
model_kategori = model_from_json(model_json_kategori)
model_kategori.load_weights('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_kategori_v3.h5')

json_file_sentimen = open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_sentimen_v3.json', 'r')
model_json_sentimen = json_file_sentimen.read()
json_file_sentimen.close()
model_sentimen = model_from_json(model_json_sentimen)
model_sentimen.load_weights('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_model_sentimen_v3.h5')

with open('D:/SKRIPSI/Program Separated/Training Models/Model Saved/saved_tokenize_v3.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)

class Classification:
	def __init__(self,text):
		self.text = text

	def klasifikasi(self):
		text = self.text
		data=[text]

		data = np.array(data)
		
		def case_folding(tokens): 
			return tokens.lower()  

		data_casefolding=[]
		for i in range(0, len(data)):
		   data_casefolding.append(case_folding(data[i]))
	

		def remove_punct(text):  
			text_nopunct = ''
			text_nopunct = re.sub('['+string.punctuation+']', ' ', text)
			return text_nopunct

		data_remove=[]
		for i in range(0, len(data)):
		   data_remove.append(remove_punct(data_casefolding[i]))

		def remove_num(text):  
			text_nonum = ''
			text_nonum = re.sub(r'\d+',' ', text)
			return text_nonum

		data_remove_num=[]
		for i in range(0, len(data)):
		   data_remove_num.append(remove_num(data_remove[i]))

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

		slang=[]
		for i in range(0, len(data)):
		   slang.append(slangword(data_remove_num[i]))

		kamus_negasi=open_kamus_prepro('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Kamus Preprocessing/Kamus negation_word.txt')

		def ganti_negasi(w):
			w_splited = w.split(' ')
			if 'tidak' in w_splited:
				index_negasi = w_splited.index('tidak')
				for i,k in enumerate(w_splited):
					if k in kamus_negasi and w_splited[i-1] == 'tidak':
						w_splited[i] = kamus_negasi[k]
			return ' '.join(w_splited)

		negasi=[]
		for i in range(0, len(data)):
		  negasi.append(ganti_negasi(slang[i]))


		tokens = [word_tokenize(sen) for sen in negasi]

		kamus_stopword=[]
		with open('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Kamus Preprocessing/Kamus stopword.txt','r') as file :
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

		filtered_words = [remove_stop_words(sen) for sen in tokens]

		stem=[]
		for i in range(0, len(data)):
		   stem.append(stemming(filtered_words[i]))
		print(stem)


		MAX_SEQUENCE_LENGTH = 50
		EMBEDDING_DIM = 300		


		prepro = [' '.join(sen) for sen in stem]
		sequences = tokenizer.texts_to_sequences(stem)

		word_index = tokenizer.word_index
		print('Number of Unique Tokens',len(word_index))

		test_cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
		x_test = test_cnn_data

		print(x_test)

		prediksi_kategori = model_kategori.predict(x_test, batch_size=1, verbose=1)
		

		prediksi_sentimen = model_sentimen.predict(x_test, batch_size=1, verbose=1)

		predict_sentimen_positif=prediksi_sentimen[0][0]
		predict_sentimen_negatif=prediksi_sentimen[0][1]


		predict_DayaTarik=prediksi_kategori[0][0]
		predict_Aksesbilitas=prediksi_kategori[0][1]
		predict_Kebersihan=prediksi_kategori[0][2]
		predict_Fasilitas=prediksi_kategori[0][3]


		class_category = ['DAYA TARIK', 'AKSESBILITAS','KEBERSIHAN','FASILITAS']
		class_sentimen = ['POSITIF', 'NEGATIF']


		for i in range(prediksi_kategori.shape[0]):
			print(str(i) + " " + class_sentimen[prediksi_sentimen[i].argmax()] + " , " + class_category[prediksi_kategori[i].argmax()] + " : " + data[i])
			hasil_sentimen = class_sentimen[prediksi_sentimen[i].argmax()]
			hasil_kategori = class_category[prediksi_kategori[i].argmax()]


		return (hasil_sentimen,hasil_kategori,prepro[0],predict_sentimen_positif,predict_sentimen_negatif,predict_DayaTarik,predict_Aksesbilitas,predict_Kebersihan,predict_Fasilitas)
			






