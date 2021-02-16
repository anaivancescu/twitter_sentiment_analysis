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
    f = open("pred_ MultinomialNB.csv", "w")
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


# 0.858 urcata pe kaggles --------------------------
#0.75126 cred, trebuie verificat  ----> nu e asta, trebuie sa caut
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

#------------------------------------------------------

# #0.886 ---> 0.894 daca n am stop_words ----------------------
# # # 0.75799 urcat pe kaggle
# clf = svm.SVC(kernel='linear', C=1.0, gamma='auto')


def corpus_to_tdidf(corpus, vocabulary):
    corpus_stemmed = [stem_tokens(tokenize(line)) for line in corpus]
    tfidfconverter = TfidfVectorizer(
        vocabulary=vocabulary, 
        min_df=0.001, 
        sublinear_tf=True)
    X = tfidfconverter.fit_transform(corpus_stemmed)
    results = X.toarray()
    vocabulary_r = tfidfconverter.get_feature_names()
    return results, vocabulary_r


# def corpus_to_tdidf(corpus, vocabulary):
#     corpus_stemmed = [stem_tokens(tokenize(line)) for line in corpus]
#     VectorizerConverter = CountVectorizer(vocabulary=vocabulary, min_df=0.001)
#     X = VectorizerConverter.fit_transform(corpus_stemmed)
#     results = X.toarray()
#     vocabulary_r = VectorizerConverter.get_feature_names()
#     return results, vocabulary_r  

def get_representation(vocabulary, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    wrd: @  che  .   ,   di  e
    idx: 0   1   2   3   4   5
    '''
    most_comm = vocabulary.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for i, iterator in enumerate(most_comm):
        idx2wd[i] = iterator[0]
        wd2idx[iterator[0]] = i
    # print(most_comm)
    return wd2idx, idx2wd

def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for i in corpus:
        tokens = tokenize(i)
        stemmed_text = stem_tokens(tokens)
        counter.update(stemmed_text)

    return counter


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


clf.fit(train, y_train)
x=clf.predict(valid)
# scor = tls.computeScore(y_valid, x)
scor = accuracy_score(y_valid, x)
scores.append(scor)
print("computed score "+str(scor))

x=clf.predict(test_data)

for i, eticheta in enumerate(x):
        predictii.append((i+5001, eticheta))

write_prediction(predictii)
