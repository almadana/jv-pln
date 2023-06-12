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
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import RobertaModel, AutoTokenizer
from os import listdir
from scipy.io import savemat, loadmat
import numpy as np
import process_textGrid as tg
from os.path import join

#load RoBERTa model
modelname = 'roberta-base' 

roberta= RobertaModel.from_pretrained(modelname)
tokenizer= AutoTokenizer.from_pretrained(modelname)
roberta.eval()
cos = nn.CosineSimilarity()

#%% main processing function, get predicted embedding for each word
#compare the predicted embedding for final token at the end of the sentence, with actual encoded vector
# sentences: list of sentences. each sentence should be a list of words
# sliding: if 0, attempts to use every previous word as context for word i. if not, use only the indicated number of previous words
def processSentence(sentence,sliding=0):
    word_cosines = []
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
        masked_word = sentence[index+1] #word that got masked (final word in sentence up to now)
        
        #print(sentence_up_to_mask)
        
        #Tokenize the sentence up to mask
        tokens = torch.tensor(tokenizer(sentence_up_to_mask)['input_ids'])
        
        #tokenize the ground truth
        target_tokens = torch.tensor(tokenizer(sentence_up_to_now)['input_ids'])
        
        print(tokens.shape)
        
        #where in each token batch is each mask?
        row,mask_index = (tokens == tokenizer.mask_token_id).nonzero(as_tuple=True)
        
        #get list of ground truth token values for each mask position - this should work!
        target_tokens_in_mask = target_tokens[0,mask_index]
        print("Target tokens in mask")
        print(target_tokens_in_mask)
        print(target_tokens)
        #alternative
        #use masked_word

        # get embeddings for mask
        predicted_embeddings = roberta(tokens)[0]
        #keep only the prediction at mask
        predicted_vector_at_mask = predicted_embeddings[row,mask_index,:]

        # get embeddings for ground truth
        target_embeddings = roberta(target_tokens)[0]
        #keep only the prediction at current token
        embedding_target_vector = target_embeddings[row,mask_index,:]




        #apply softmax
        cosines = cos(predicted_vector_at_mask,embedding_target_vector)
        word_cosines.append(cosines.item())
        #for each word, embed the whole sentence up to that word, and extract the last vector (ignore sentencestart and finish tokens)        
    return(word_cosines)



#%%  get Transcripts and process them with CamemBERT
paris_path = '/DATA1/Dropbox/PhD/Project ECOSud/JulesVerne/jv-pln/data'
#process_textGrid.py
textGrid_folder = "../data/wav/revised/"
#textGrid_folder = join(paris_path,'wav','revised/')

files = listdir(textGrid_folder)


filenames = np.array(list((filter(lambda x: x.endswith(".TextGrid") , files))),dtype=object)
filenames.sort()
phonemes = np.empty(filenames.shape,dtype="object")
words = np.empty(filenames.shape,dtype="object")
word_markings=np.empty(filenames.shape,dtype="object")
words=np.empty(filenames.shape,dtype="object")
phon_markings=np.empty(filenames.shape,dtype="object")


sliding= 420#sliding parameter. if 0, will use ALL previous context... don't think it can handle more than 100... see CamemBERT documentation for max sentence size...

xmin_all = ()
xmax_all = ()
text_markings_all = ()

for i,file in enumerate(filenames):
    all_markings = tg.parseTextGrid(textGrid_folder+file)
    xmin,xmax,text_markings = zip(*all_markings[0]) #[0] text markings, [1] phoneme markings
    text_markings_all = text_markings_all + text_markings
    xmin_all = xmin_all + xmin
    xmax_all = xmax_all + xmax
    
    xmin,xmax,p_markings = zip(*all_markings[1]) #[0] text markings, [1] phoneme markings
    xminf = [float(x) for x in xmin]
    xmaxf = [float(x) for x in xmax]
    phonemes[i]=p_markings
    phon_markings[i]=np.array([xminf,xmaxf]).transpose()
    
probs = processSentence(np.array(text_markings_all),sliding)
xminf = [float(x) for x in xmin_all]
xmaxf = [float(x) for x in xmax_all]
probsf= [float(x) for x in probs]
word_markings = np.array([xminf,xmaxf,probsf]).transpose()
words = text_markings_all

word_trial = np.empty(filenames.shape,dtype="object")
word_markings_trial = np.empty(filenames.shape,dtype="object")

offsets = np.where(np.diff(xmaxf)<-30)[0]
offsets=offsets+1
offsets=np.insert(offsets,0,0)   

for idx, off in zip(np.arange(0,len(filenames)),offsets):
    if idx ==len(filenames)-1:
        word_markings_trial[idx] = word_markings[off:]
        word_trial[idx] = text_markings_all[off:]
        
    # elif idx == len(filenames)-1:
    #     word_markings_trial[idx] = probsf[off+1:]
    #     word_trial[idx] = text_markings_all[off+1:]
        
    else:
        word_markings_trial[idx] = word_markings[off:offsets[idx+1]]
        word_trial[idx] = text_markings_all[off:offsets[idx+1]]


            
#markings = tg.parseTextGrid("../../transcriptions/renard1.TextGrid") #list, first  [0] are text markings, second [1] are phonetic markings
# text_markings = [m[0] for m in markings]
# phon_markings = [m[1] for m in markings]
# xmin,xmax,tmarkings = zip(*text_markings)

# savemat("../data/embed.mat", {"word_markings":word_markings,"words":words,"phon_markings":phon_markings,"phonemes":phonemes,"filenames":filenames})
savemat(join(paris_path,'embed_420_mask_cos.mat'), {"word_markings":word_markings_trial,"words":word_trial,"phon_markings":phon_markings,"phonemes":phonemes,"filenames":filenames})

#markings_results = processSentences(tmarkings,sliding=50)
#output = list(zip(xmin,xmax,tmarkings,markings_results[0]))
#fxmin,fxmax,fmarkings = zip(*text_markings)
