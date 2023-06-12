#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 04:48:53 2023

Main script that uses a BERT transformer to obtain predictability values for each word in a text sequence.

First block imports utilities and the CamemBERT model and tokenizer

Second block defines the main function that processes sentences. Sentences are supposed to be lists of words. You can pass an array with many sentences. Each sentence will be trated independently.
The output is a list of predictability (kinda like probabilities, but not quite) levels for each word in the sentence. Actually it's a list of lists, one list for each sentence.

There is a block for model validation. See "validation_set.py" for details.

@author: Álvaro Cabana
"""

#%%  IMPORTS and model loading

import torch
import matplotlib.pyplot as plt
from transformers import RobertaForMaskedLM, AutoTokenizer
from os import listdir
from scipy.io import savemat, loadmat
import numpy as np
import process_textGrid as tg
from os.path import join

#load RoBERTa model
modelname = 'roberta-base' 

roberta= RobertaForMaskedLM.from_pretrained(modelname)
tokenizer= AutoTokenizer.from_pretrained(modelname)
roberta.eval()


#%% main processing function, get predicted probabilities for each word
#compare the predicted vector for final token at the end of the sentence, with actual encoded vector
# sentences: list of sentences. each sentence should be a list of words
# sliding: if 0, attempts to use every previous word as context for word i. if not, use only the indicated number of previous words
def processSentence(sentence,sliding=0):
    word_probs = []
    for index,word in enumerate(sentence):
        #nTokens_per_word = len(torch.tensor(tokenizer.encode(word))) - 2 #if only one token for this word, this should be 1
        
        #if sliding is nonzero, only that number of preceding words will be used to build the sentence
        if sliding==0:
            firstWordIndex=0
        else:
            firstWordIndex=max(0,index-sliding)
        
        #actual sentence up to now
        sentence_up_to_now = [" ".join(sentence[firstWordIndex:(index+1)])]
        #sentence_up_to_now += " ,"
        #sentence with lacking current word (replaced with mask)
        sentence_up_to_mask = " ".join(sentence[firstWordIndex:(index)])
        sentence_up_to_mask = [sentence_up_to_mask + " <mask>"]
        #print(sentence_up_to_mask)
        tokens = torch.tensor(tokenizer(sentence_up_to_mask)['input_ids'])
        #tokenize the ground truth
        target_tokens = torch.tensor(tokenizer(sentence_up_to_now)['input_ids'])
        print(tokens.shape)
        #where in each token batch is each mask?
        row,mask_index = (tokens == tokenizer.mask_token_id).nonzero(as_tuple=True)
        #get list of ground truth token values for each mask position
        target_tokens_in_mask = target_tokens[0,mask_index]

        #predict tokens
        predicted_tokens = roberta(tokens)[0]
        #keep only the prediction at mask
        predicted_vectors = predicted_tokens[row,mask_index,:]

        #apply softmax
        probs = predicted_vectors.softmax(dim=1)
        probs_targets = torch.tensor([ probs[i,j] for i,j in zip(row,target_tokens_in_mask)])
        print(word+" "+str(probs_targets))
        maxProb = torch.max(probs)
        probs_norm = probs_targets/maxProb
        word_probs.append(probs_targets.item())
        #for each word, embed the whole sentence up to that word, and extract the last vector (ignore sentencestart and finish tokens)        
    return(word_probs)



#%%  get Transcripts and process them with CamemBERT
paris_path = '/DATA1/Dropbox/PhD/Project ECOSud/JulesVerne/jv-pln/data'
#process_textGrid.py
textGrid_folder = "../data/wav/revised/"
textGrid_folder = join(paris_path,'wav','revised/')

files = listdir(textGrid_folder)


filenames = np.array(list((filter(lambda x: x.endswith(".TextGrid") , files))),dtype=object)
filenames.sort()
phonemes = np.empty(filenames.shape,dtype="object")
words = np.empty(filenames.shape,dtype="object")
word_markings=np.empty(filenames.shape,dtype="object")
words=np.empty(filenames.shape,dtype="object")
phon_markings=np.empty(filenames.shape,dtype="object")


sliding=50 #sliding parameter. if 0, will use ALL previous context... don't think it can handle more than 100... see CamemBERT documentation for max sentence size...


for i,file in enumerate(filenames):
    all_markings = tg.parseTextGrid(textGrid_folder+file)
    xmin,xmax,text_markings = zip(*all_markings[0]) #[0] text markings, [1] phoneme markings
    probs = processSentence(np.array(text_markings),sliding)
    xminf = [float(x) for x in xmin]
    xmaxf = [float(x) for x in xmax]
    probsf= [float(x) for x in probs]
    word_markings[i]=np.array([xminf,xmaxf,probsf]).transpose()
    words[i]=text_markings
    xmin,xmax,p_markings = zip(*all_markings[1]) #[0] text markings, [1] phoneme markings
    xminf = [float(x) for x in xmin]
    xmaxf = [float(x) for x in xmax]
    phonemes[i]=p_markings
    phon_markings[i]=np.array([xminf,xmaxf]).transpose()
#markings = tg.parseTextGrid("../../transcriptions/renard1.TextGrid") #list, first  [0] are text markings, second [1] are phonetic markings
# text_markings = [m[0] for m in markings]
# phon_markings = [m[1] for m in markings]
# xmin,xmax,tmarkings = zip(*text_markings)

# savemat("../data/embed.mat", {"word_markings":word_markings,"words":words,"phon_markings":phon_markings,"phonemes":phonemes,"filenames":filenames})
savemat(join(paris_path,'embed.mat'), {"word_markings":word_markings,"words":words,"phon_markings":phon_markings,"phonemes":phonemes,"filenames":filenames})

#markings_results = processSentences(tmarkings,sliding=50)
#output = list(zip(xmin,xmax,tmarkings,markings_results[0]))
#fxmin,fxmax,fmarkings = zip(*text_markings)
