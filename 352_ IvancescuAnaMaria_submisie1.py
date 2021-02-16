import nltk
nltk.download('wordnet')
import pandas as pd
import numpy as np
import random as random
import matplotlib 
from collections import Counter
from sklearn import svm
import re
from nltk.stem import SnowballStemmer
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('italian'))

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("italian",ignore_stopwords=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

TRAIN_FILE = ''
TEST_FILE = ''
TXT_COL = 'text'
LBL_COL = 'label'

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


def write_prediction( predictions):
    print("Gata")
    f = open("pred_MultinomialNB1.csv", "w")
    f.write("id,label\n")
    for prediction in predictions:
        # (id,label ul prezis)
        f.write(str(prediction[0]) + "," + str(prediction[1]) + "\n")    

def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    text_fara_spatii = text.rstrip() 
    text = re.sub('[^A-Za-z!]+', ' ', text_fara_spatii)
    return nltk.WordPunctTokenizer().tokenize(text) 

def stem_tokens(tokens):
    stemmed_tokens=[stemmer.stem(token) for token in tokens]
    text_stemmed = " ".join(stemmed_tokens) #imi construieste o prop cu spatii intre ele
    return text_stemmed


# 0.858 urcata pe kaggle
#0.75126 ----->0.77935
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


def corpus_to_tdidf(corpus, vocabulary):
    corpus_stemmed = [stem_tokens(tokenize(line)) for line in corpus]
    tfidfconverter = TfidfVectorizer(
        vocabulary=vocabulary, 
        min_df=0.001, 
        sublinear_tf=True)
    X = tfidfconverter.fit_transform(corpus_stemmed)
    results = X.toarray()
    #print(results)
    vocabulary_r = tfidfconverter.get_feature_names()
    return results, vocabulary_r



corpus = train_df['text']
labels = train_df['label']
predictii = []
scores = []
x=[]

data, vocabulary = corpus_to_tdidf(corpus, None)
print("Vocabular" + str(vocabulary))
print("Vocabulary length: " + str(len(vocabulary)))
test_data, vocabulary = corpus_to_tdidf(test_df['text'], vocabulary)

train, valid, y_train, y_valid = train_test_split(data, labels, test_size=0.2,  random_state=20)

# def cross_validate(data,labels,k):
#     chunk_size = (len(labels) // k)
#     indici = np.arange(0, len(labels))
#     random.shuffle(indici)
#     for i in range(0, len(labels), chunk_size):
#         valid_indici = indici[i:i+chunk_size]
#         train_indici = np.concatenate([indici[0:i], indici[i+chunk_size:]])
#         train = data[train_indici]
#         valid=data[valid_indici]
#         y_train=labels[train_indici]
#         y_valid=labels[valid_indici]
#         yield train, valid, y_train, y_valid

# for train, valid, y_train, y_valid in cross_validate(data, labels, 10):
#     clf.fit(train, y_train)
#     x=clf.predict(valid)
#     scor = accuracy_score(y_valid, x)
#     scores.append(scor)
#     print("computed score "+str(scor))


# start_time=time.time()


clf.fit(train, y_train)
x=clf.predict(valid)

print(confusion_matrix(y_valid, 
[labels_bun for i, labels_bun in enumerate(x)], normalize='true'))

scor = accuracy_score(y_valid, x)
scores.append(scor)
print("computed score "+str(scor))
x=clf.predict(test_data)

for i, labels_bun in enumerate(x):
        predictii.append((i+5001, labels_bun))
       
write_prediction(predictii)



# print("--- %s seconds ---" % (time.time() - start_time))