#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 05:52:57 2022

@author: perezoso
"""

#Convert output .json files from Google Cloud (transcriptions) to plain .txt files

from os import listdir
import re
import json

json_re = re.compile(".*\.json$")
# extract_re = re.compile('"transcript":.*',re.MULTILINE)
# clean_re = re.compile('\s+"transcript":\s+"([^"]*)"',re.MULTILINE)

folder = '../data/wav/json'

files = listdir(folder)

for file in files:
    is_jason = json_re.match(file)
    if is_jason:
        filename = re.sub("\.wav.*$","",file)
        print(filename)
        jsonFile = json.load(open(folder+"/"+file))
        transcript = jsonFile['text']
        transcript = re.sub("[.,:;\"]","",transcript).lower()
        print(transcript)
        out_file = open(folder+"/"+filename+".txt",'w')
        out_file.write(transcript)
        out_file.close()
#        print(jsonFile)
        # extracted = extract_re.findall(jsonFile)
        # extracted = " ".join(extracted)
        # cleaned = clean_re.search(extracted)
        # print(cleaned.group(1))
        # #print(cleaned)

        
