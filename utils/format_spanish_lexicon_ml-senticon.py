#!/usr/bin/python3
# this script formats the ML-SentiCon lexicon for Spanish

import sys
import xml.etree.ElementTree as ET


# source data file
ifile = sys.argv[1]

# output data file
ofile = sys.argv[2]

with open(ofile, 'w') as fout:
    tree = ET.parse(ifile)
    root = tree.getroot()
    for lemma in root.iter('lemma'):
        if lemma.text.strip().isalpha():
            word = lemma.text.strip()
            wpol = lemma.attrib['pol']
            fout.write(str(word) + '\t' + str(wpol) + '\n')

