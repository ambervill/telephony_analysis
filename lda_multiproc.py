#!/usr/bin/env python
# coding: utf-8


import os
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
import pandas as pd
import numpy as np
import multiprocessing 


TRANSCRIBE_PATH = "/media/storage/alekseychuk/LDA/texts"

data_words = []

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


for folder in os.listdir(TRANSCRIBE_PATH):
    if not folder.startswith("."):
        print("folder: " + folder)
        folder_path = TRANSCRIBE_PATH + '/' + folder
        #print("path: " + folder_path)
        for text_name in os.listdir(folder_path):
            #print("text_name: " + text_name)
            text_path = folder_path + '/' + text_name
            #print("text_path: " + text_path)
            f = open(text_path, 'r')
            words = f.read()
            if words:
                data_words.append(words.split())
                
                
for j, text in enumerate(data_words):
    for i, word in enumerate(data_words[j]):
        if word == 'диадох' or word == 'виадук':
            data_words[j][i] = 'диадок'

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

bigram_mod = gensim.models.phrases.Phraser(bigram)


data_words_bigrams = make_bigrams(data_words)

nlp = spacy.load('ru_core_news_sm', disable=['parser', 'ner'])


# data_lemmatized = []
def lemmatization(texts):
    output = []
    print('lemmatization started. len=', len(texts))
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    with open("data_lemmatized.txt", "a") as txt_file:
        for line in output:
            txt_file.write(" ".join(line) + "\n")


# lemmatization(data_words_bigrams, data_lemmatized)

length = len(data_words_bigrams)

 
ranges = np.linspace(0,length,8)

if __name__ == '__main__':
    processes = []

    p1 = multiprocessing.Process(target=lemmatization, args=(data_words_bigrams[:int(ranges[1])],))
    processes.append(p1)
    p1.start()
    print('Process 1 started.')
    
    p2 = multiprocessing.Process(target=lemmatization, args=(data_words_bigrams[int(ranges[1]):int(ranges[2])],))
    processes.append(p2)
    p2.start()
    print('Process 2 started.')
    
    p3 = multiprocessing.Process(target=lemmatization, args=(data_words_bigrams[int(ranges[2]):int(ranges[3])],))
    processes.append(p3)
    p3.start()
    print('Process 3 started.')
    
    p4 = multiprocessing.Process(target=lemmatization, args=(data_words_bigrams[int(ranges[3]):int(ranges[4])],))
    processes.append(p4)
    p4.start()
    print('Process 4 started.')
    
    p5 = multiprocessing.Process(target=lemmatization, args=(data_words_bigrams[int(ranges[4]):int(ranges[5])],))
    processes.append(p5)
    p5.start()
    print('Process 5 started.')
    
    p6 = multiprocessing.Process(target=lemmatization, args=(data_words_bigrams[int(ranges[5]):int(ranges[6])],))
    processes.append(p6)
    p6.start()
    print('Process 6 started.')
    
    p7 = multiprocessing.Process(target=lemmatization, args=(data_words_bigrams[int(ranges[6]):],))
    processes.append(p7)
    p7.start()
    print('Process 7 started.')
    
    
    for process in processes:
        process.join()
