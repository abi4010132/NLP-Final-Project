import pickle, gzip
import pandas as pd
import numpy as np

import nltk
nltk.download('wordnet')

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.nist_score import corpus_nist

from nltk.translate import meteor

#loading the pickle file with results, and preprocesing them so they can be used by the methods for evaluation
#basline data
df_baseline = pd.read_pickle('results_baseline.pickle')
df_baseline.pop("Unnamed: 0")
df_baseline.pop("Question")
for index, row in df_baseline.iterrows():
    row['Answer']=[row['Answer']]
#attention model data
df_attention = pd.read_pickle('results_attention.pickle')
df_attention.pop("Unnamed: 0")
df_attention.pop("Question")
for index, row in df_attention.iterrows():
    row['Answer']=[row['Answer']]

#helper functions for the individual metrics
def blue_score(references, hypothesis):
    score = sentence_bleu(references, hypothesis,(0.5, 0.5, 0, 0)) 
    return(score)
def nist_score(references, hypothesis):
    score = sentence_nist(references, hypothesis,2)
    return(score)
def meteor_score(references, hypothesis):
    score = meteor(references, hypothesis) 
    return(score)
def corpus_meteor(expected, predicted):
    meteor_score_sentences_list = list()
    [meteor_score_sentences_list.append(meteor_score(expect, predict)) for expect, predict in zip(expected, predicted)]
    meteor_score_res = np.mean(meteor_score_sentences_list)
    return meteor_score_res

#corpus scores for baseline
print("Baseline model")
blueScore= corpus_bleu(df_baseline['Answer'].tolist(),df_baseline['Predicted Answer Baseline'].tolist())
print("blue> ", blueScore)
nistScore= corpus_nist(df_baseline['Answer'].tolist(),df_baseline['Predicted Answer Baseline'].tolist(),n=2)
print("nist> ", nistScore)
meteorScore= corpus_meteor(df_baseline['Answer'],df_baseline['Predicted Answer Baseline'])
print("meteor> ", meteorScore)

#corpus scores for attention
print("Attention model")
blueScore= corpus_bleu(df_attention['Answer'].tolist(),df_attention['Predicted Answer'].tolist())
print("blue> ", blueScore)
nistScore= corpus_nist(df_attention['Answer'].tolist(),df_attention['Predicted Answer'].tolist(),n=2)
print("nist> ", nistScore)
meteorScore= corpus_meteor(df_attention['Answer'],df_attention['Predicted Answer'])
print("meteor> ", meteorScore)
