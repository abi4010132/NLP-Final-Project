import pandas as pd
import numpy as np

import nltk
nltk.download('wordnet')

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.nist_score import corpus_nist
from nltk.translate import meteor

def read_data():
    df = pd.read_pickle('results_baseline.pickle')
    df.pop("Unnamed: 0")
    df.pop("Question")
    for index, row in df.iterrows():
        row['Answer']=[row['Answer']]
    return df

def blue_score(references, hypothesis): #both inputs are list of words(tokens)
    score = sentence_bleu(references, hypothesis,(0.5, 0.5, 0, 0)) # we can change the weights=(1, 0, 0, 0) if we want to prioritize different n-grams
    return(score)
def nist_score(references, hypothesis):
    score = sentence_nist(references, hypothesis,2) # we can change the n=... if we want to prioritize different n-grams
    return(score)
def meteor_score(references, hypothesis):
    score = meteor(references, hypothesis) # we can change the n=... if we want to prioritize different n-grams
    return(score)
def corpus_meteor(expected, predicted):
    meteor_score_sentences_list = list()
    [meteor_score_sentences_list.append(meteor_score(expect, predict)) for expect, predict in zip(expected, predicted)]
    meteor_score_res = np.mean(meteor_score_sentences_list)
    return meteor_score_res

def main():
    df = read_data()
    blueScore= corpus_bleu(df['Answer'].tolist(),df['Predicted Answer Baseline'].tolist())
    print("blue> ", blueScore)
    nistScore= corpus_nist(df['Answer'].tolist(),df['Predicted Answer Baseline'].tolist(),n=4)
    print("nist> ", nistScore)
    meteorScore= corpus_meteor(df['Answer'],df['Predicted Answer Baseline'])
    print("meteor> ", meteorScore)

    blues=0
    nists=0
    mateors=0
    for index, row in df.iterrows():
        blues+=blue_score(row['Answer'], row['Predicted Answer Baseline'])
        nists+=nist_score(row['Answer'], row['Predicted Answer Baseline'])
        mateors+=meteor_score(row['Answer'], row['Predicted Answer Baseline'])
    blues=blues/len(df['Answer'])
    nists=nists/len(df['Answer'])
    mateors=mateors/len(df['Answer'])
    print("-------")
    print("blue> ", blues)
    print("nist> ", nists)
    print("meteor> ", mateors)

if __name__ == "__main__":
    main()