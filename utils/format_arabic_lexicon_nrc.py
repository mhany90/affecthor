#!/usr/bin/python3
# this file correctly formats the NRC emotion lexicon for Arabic
# entries that contain the same word in the first field are merged

import sys
import string

# source data file
ifile = sys.argv[1]

# output data file
ofile = sys.argv[2]

# dictionary that maps a word to its emotion indicators
emoword = {}

with open(ifile, 'r') as fin:
    lines = fin.readlines()
    header = lines[0]

    for line in lines[1:]:
        word = line.split()[0]
        emos = line.split()[1:]

        # collect and merge emotions
        if word not in emoword:
            emoword[word] = emos
        else:
            merged = list(map(lambda y: max(y), zip(emoword[word], emos)))
            emoword[word] = merged

with open(ofile, 'w') as fout:
    fout.write(header)
    for word in emoword:
        sout = word + '\t'
        sout += '\t'.join(emoword[word])
        fout.write(sout + '\n')

