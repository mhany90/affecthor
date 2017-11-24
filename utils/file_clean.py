#!/usr/bin/python3
# this script deals with certain characters and words from a tab-separated field in a file
# it is meant to be used before tokenization

import re
import sys
import string

# source data file
ifile = sys.argv[1]

# language of the source data file
lang = sys.argv[2]

# field to filter
filter_field = int(sys.argv[3])

# output data file
ofile = sys.argv[4]

# matches repeated characters
re_repeat = r'([\D])\1\1+'

# matches twitter mentions @username
re_mention = r'@([A-Za-z0-9_]+)'

# matches control characters
re_control = r'(\\[tnrfv])+'

# matches other special characters such as emoticons
re_emoticon = r'([^\w\s' + string.punctuation + r'])'

with open(ifile, 'r') as fin:
    with open(ofile, 'w') as fout:
        for line in fin.readlines():
            ind = 0
            fields = line.split('\t')

            for field in fields:
                ind += 1
                # directly output or filter a field
                if ind != filter_field:
                    fout.write(field)
                else:

                    # Arabic data
                    if lang == 'Ar':
                        # delete quotation marks to avoid weka errors
                        a = field.replace('"', '')
                        # replace mentions with a generic token
                        b = re.sub(re_mention, r'@username', a)
                        # replace control characters
                        c = re.sub(re_control, r' ', b)
                        # replace emoticons
                        d = re.sub(re_emoticon, r' \1 ', c)
                        fout.write(d.lower())

                    # Spanish data
                    elif lang == 'Es':
                        # replace repeated letters with only 2 occurrences
                        a = re.sub(re_repeat, r'\1\1', field)
                        # replace mentions with a generic token
                        b = re.sub(re_mention, r'@username', a)
                        # replace control characters
                        c = re.sub(re_control, r' ', b)
                        # replace emoticons
                        d = re.sub(re_emoticon, r' \1 ', c)
                        fout.write(d.lower())

                    # English data
                    else:
                        # replace repeated letters with only 2 occurrences
                        a = re.sub(re_repeat, r'\1\1', field)
                        # replace mentions with a generic token
                        b = re.sub(re_mention, r'@username', a)
                        # replace control characters
                        c = re.sub(re_control, r' ', b)
                        # replace emoticons
                        d = re.sub(re_emoticon, r' \1 ', c)
                        fout.write(d.lower())

                # print separators
                if ind < len(fields):
                    fout.write('\t')
                else:
                    fout.write('')

