import os
import pandas as pd
import numpy as np
import pickle
import warnings
from multiprocessing import Process, Lock
import tensorflow_hub as hub
import tensorflow_text

TRANSCRIBE_PATH = "/media/storage/alekseychuk/LDA/recognition"

data_strings = []

for folder in os.listdir(TRANSCRIBE_PATH):
    if folder.startswith("."):
        continue
    print("folder: " + folder)
    folder_path = TRANSCRIBE_PATH + '/' + folder
    for text_name in os.listdir(folder_path):
        if text_name.startswith("."):
            continue
        text_path = folder_path + '/' + text_name
        f = open(text_path, 'r')
        words = f.read()
        if words:
            split = words.split()
            if len(split) > 120:
                data_strings.append(words)
                    
                    
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
corpus_embeds = []
counter = 0
for string in data_strings:
    corpus_embeds.append(embed(string)[0])
    counter += 1
    if counter % 100 == 0:
        print("processed document " + str(counter))

with open('corpus_embeds.pkl', 'wb') as f:
    pickle.dump(corpus_embeds, f)