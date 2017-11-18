#!/usr/bin/python3
# this script extracts from and formats the Spanish lexicons

import sys
import xml.etree.ElementTree as ET


# source data file
ifile = sys.argv[1]

# output data file
ofile = sys.argv[2]

with open(ofile, 'w') as fout:

    # rule for xml lexicons
    if ifile.endswith('.xml'):
        tree = ET.parse(ifile)
        root = tree.getroot()
        for lemma in root.iter('lemma'):
            fout.write(lemma.text.strip() + '\t' + str(lemma.attrib['pol']) + '\n')

    # assume other lexicons are already in the correct format
    else:
        for line in open(ifile, 'r').readlines():
            fout.write(line)

