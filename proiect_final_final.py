import math
import random
import re
from collections import Counter

import matplotlib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


nltk.download("wordnet")


stop_words = set(stopwords.words("italian"))


stemmer = SnowballStemmer("italian", ignore_stopwords=True)


TRAIN_FILE = ""
TEST_FILE = ""
TXT_COL = "text"
LBL_COL = "label"

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


def write_prediction(predictions):
    print("Gata")
    f = open("pred_MultinomialNBceva.csv", "w")
    f.write("id,label\n")
    for prediction in predictions:
        f.write(str(prediction[0]) + "," + str(prediction[1]) + "\n")


def tokenize(text):
    """Generic wrapper around different tokenization methods."""
    text_fara_spatii = text.rstrip()
    text = re.sub("[^A-Za-z!]+", " ", text_fara_spatii)
    return nltk.WordPunctTokenizer().tokenize(text)


def stem_tokens(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    text_stemmed = " ".join(stemmed_tokens)
    return text_stemmed


# #VECTORIZAREA

# #0.881
# clf = svm.NuSVC( gamma='auto')

# # 0.886 ---> 0.89 daca n am stop_words
# clf_local = svm.SVC(kernel="linear", C=1.0)
# clf_kaggle = svm.SVC(kernel="linear", C=1.0)


# #  0.899-->1    0.74449  urcaa pe kaggle  
# from sklearn.linear_model import SGDClassifier
# clf_local = SGDClassifier(loss="log", penalty="l2", random_state=1,learning_rate='constant',eta0=1,shuffle=True)
# clf_kaggle = SGDClassifier(loss="log", penalty="l2", random_state=1,learning_rate='constant',eta0=1,shuffle=True)

from sklearn.naive_bayes import BernoulliNB
clf_local = BernoulliNB()
clf_kaggle = BernoulliNB()



def corpus_to_tdidf(corpus, vocabulary, min_df=0):
    corpus_stemmed = [stem_tokens(tokenize(line)) for line in corpus]
    tfidfconverter = TfidfVectorizer(
        vocabulary=vocabulary, min_df=min_df, sublinear_tf=True
    )
    X = tfidfconverter.fit_transform(corpus_stemmed)
    results = X.toarray()
    vocabulary_r = tfidfconverter.get_feature_names()
    return results, vocabulary_r


corpus = train_df["text"]
labels = train_df["label"]
predictii = []
scores = []
x = []


# Aici facem pentru local
data, local_vocabulary = corpus_to_tdidf(corpus, None, 0.001)
# print("Vocabular local:" + str(local_vocabulary))
# print("Lungime vocabular local: " + str(len(local_vocabulary)))


train, valid, y_train, y_valid = train_test_split(
    data, labels, test_size=0.2, random_state=20
)


clf_local.fit(train, y_train)
x = clf_local.predict(valid)
scor = accuracy_score(y_valid, x)
scores.append(scor)

print("local computed score " + str(scor))
# -------------------------------------

# Facem pentru Kaggle
# returnez vocabularul de pe kaggle
test_data, kaggle_vocabulary = corpus_to_tdidf(test_df["text"], None, min_df=0.001)
# print("Vocabular kaggle:" + str(kaggle_vocabulary))
# print("Lungime vocabular kaggle: " + str(len(kaggle_vocabulary)))
# trec datele de train prin TDIDF specificand ca vreau sa ia vocabularul de pe kaggle
data, kaggle_vocabulary = corpus_to_tdidf(corpus, kaggle_vocabulary)

# pentru a antrena modelul nu mai este nevoie de split 80/20
# folosesc si cele 20% de date de 'test' ca sa obtin mai
# multe date de antrenament pentru model
train, valid, y_train, y_valid = train_test_split(
    data, labels, test_size=0.01, random_state=20
)

print(confusion_matrix(y_valid, 
[labels_bun for i, labels_bun in enumerate(x)], normalize='true'))
clf_kaggle.fit(train, y_train)
x = clf_kaggle.predict(test_data)
# -------------------

for i, eticheta in enumerate(x):
    predictii.append((i + 5001, eticheta))

write_prediction(predictii)