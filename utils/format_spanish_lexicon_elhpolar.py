#!/usr/bin/python3
# this script formats the ElhPolar lexicon

import sys
import string


# source data file
ifile = sys.argv[1]

# list of stopwords
stopwords = set(open(sys.argv[2], 'r').read().split())

# output data file
ofile = sys.argv[3]

# keep polarity for single words
single_pol = {}

# split expressions into words and assign the global polarity to each one
multi_pol = {}

with open(ifile, 'r') as fin:
    for line in fin.readlines():
        if not line.startswith('#'):
            fields = line.split()
            if fields:
                words = fields[:-1]
                polarity = fields[-1]

                if len(words) == 1:
                    single_pol[words[0]] = polarity
                else:
                    for w in words:
                        if w not in stopwords and len(w) > 3:
                            if w not in multi_pol:
                                multi_pol[w] = polarity
                            elif multi_pol[w] and multi_pol[w] != polarity:
                                multi_pol[w] = None

with open(ofile, 'w') as fout:
    for w in single_pol:
        fout.write(str(w) + '\t' + str(single_pol[w]) + '\n')
    for w in multi_pol:
        if multi_pol[w] and w not in single_pol:
            fout.write(str(w) + '\t' + str(multi_pol[w]) + '\n')

