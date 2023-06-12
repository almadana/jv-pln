#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 05:16:55 2022

This script defines a function that parses praat .TextGrid objects and returns a list of 3-uples containing xmin, xmax and text tags for each item[]

For the Oli stories project, item[0] should be words and item[1] phonemes

ATTENTION! Praat saves TextGrid objects in UTF-16BE encoding. Tu convert it to UTF-8 you can use iconv:
for example: iconv -f UTF-16BE -t UTF-8 renard1.TextGrid -o renard1.TextGrid_utf8 // or use batch conversion
    

@author: Álvaro Cabana
"""

import re

# find item 2 marking
marking_re = re.compile("intervals \[[0-9]*\]:.*$\n^\s*xmin = ([0-9]*\.[0-9]*).*$\n\s*xmax = ([0-9]*\.[0-9]*).*$\n\s*text = \"(.+)\"",re.MULTILINE)
items_re = re.compile("\s+item\s\[[0-9]+\]")


    #load and parse TextGrid files (Praat) and produce a string of text and timestamps
    
def parseTextGrid(fileName):
    textFile = open(fileName,'r').read()
    items = [i.end() for i in items_re.finditer(textFile)]
    items.append(len(textFile)-1)
    readings=[]
    for i in range(0,len(items)-1):
        readings.append( marking_re.findall(textFile[items[i]:(items[i+1])]) )
    return(readings)
#example marking
# "intervals [1]:
#             xmin = 0 
#             xmax = 1.3273274411213993 
#             text = """





# markings = [] #array of tuples containing xmin, xmax, marking

# for line in textFile:
#     #look for intervals markings
#     if marking.re.search(line):
#         xmin=0
#         xmax=0
#         text.marking = ""
#     else:
#         if xmin.re.search(line):
#             xmin=0