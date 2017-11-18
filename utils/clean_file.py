#!/usr/bin/python3
# this script deletes and replaces certain special characters and words

import re
import sys
import string
from nltk.corpus import stopwords

# source data file
ifile = sys.argv[1]

# language of the source data file
lang = sys.argv[2]

# output data file
ofile = sys.argv[3]


with open(ifile, 'r') as fin:
    with open(ofile, 'w') as fout:

        # delete quotation marks in Arabic
        # weka returns an error otherwise
        if lang == 'Ar':
            for line in [y.replace('"', '') for y in fin.readlines()]:
                fout.write(line)
        else:
            for line in fin.readlines():
                fout.write(line)
                
