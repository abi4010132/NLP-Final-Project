import pandas as pd
import numpy as np
import nltk
import re
import gensim
import random
import contractions
# from textblob import TextBlob
from keras.layers import Bidirectional, LSTM, Dense, Activation, dot, BatchNormalization, concatenate, Input, Embedding
from keras.models import Sequential, Model
from keras import initializers
# nltk.download('punkt')

def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', str(text))
    return text
def lower_case(text):
    return text.lower()
def remove_html(text):
    return re.sub(r'https?://[^\s/$.?#].[^\s]*', '', text,)
def remove_numbers(text):
    return re.sub(r'[0-9]+', '', text,)
def tokenize(text):
    return nltk.word_tokenize(text)
def add_sos_eos(text):
    text=["<sos>"]+text+["<eos>"]
    return text
def fix_contractions(text):
    return contractions.fix(text)

def main():
    data = pd.read_csv('data/single_qna.csv')
    data = data[['Question', 'Answer']]
    data = data.astype(str)
    for name in data.columns:
        data.loc[:,name] = data.loc[:,name].apply(remove_html)
        print("Removed HTML")
        data.loc[:,name] = data.loc[:,name].apply(remove_special_characters)
        print("Removed special chars")
        data.loc[:,name] = data.loc[:,name].apply(fix_contractions)
        print("Removed contractions")
        data.loc[:,name] = data.loc[:,name].apply(lower_case)
        print("Lower cased!")
        data.loc[:,name] = data.loc[:,name].apply(remove_numbers)
        print("Removed numbers :)")
        data.loc[:,name] = data.loc[:,name].apply(tokenize) # from now on tokens
        print("Tokenized!")
        print("Added starting and finishing tokens...")
    
    data.to_csv('data/single_qna_clean_data.csv')

if __name__ == "__main__":
    main()
