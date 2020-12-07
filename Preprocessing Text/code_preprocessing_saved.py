import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

data = pd.read_csv('D:/SKRIPSI/Dataset/datav3hasil.tsv', header = None, delimiter='\t')

data.columns = ['Id','Text', 'Label']

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

data['Text_slang'] = data['Text_Nonum'].apply(lambda x: slangword(x))
print ("slangword")

kamus_negasi=open_kamus_prepro('D:/SKRIPSI/Program Separated/Preprocessing Text/Kamus Preprocessing/Kamus negation_word.txt')

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
with open('D:/SKRIPSI/Program Separated/Preprocessing Text/Kamus Preprocessing/Kamus stopword.txt','r') as file :
	for line in file :
		slang=line.replace("'","").strip()
		kamus_stopword.append(slang)

def remove_stop_words(tokens):
    return [word for word in tokens if word not in kamus_stopword]

data['Akhir'] = [remove_stop_words(sen) for sen in tokens] 
print ("stopword")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(tokens):  
    data_stem =[]
    for i in tokens:
      kata = stemmer.stem(i)
      data_stem.append(kata)
    return data_stem

# data['Akhir'] = data['filtered_words'].apply(lambda x: stemming(x))
print ("stemming")

result = [' '.join(sen) for sen in data['Akhir']]

data['Text_Final'] = result

data['tokens'] = data['Akhir']

data = data[['Id','Text_Final', 'Label']]

data.to_csv('D:/SKRIPSI/Program Separated/Preprocessing Text/Hasil Preprocessing/preprocessing_dataset_sentimen_faizal.tsv', index=False, header=False, sep="\t")

