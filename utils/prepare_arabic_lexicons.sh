#!/bin/bash

AR_LEX_SDIR=/home/s3094723/SEMEVAL/AffectiveTweets/lexicons/ar
AR_LEX_ODIR=/home/s3094723/wekafiles/packages/AffectiveTweets/lexicons/ar

for filename in ${AR_LEX_SDIR}/*.txt; do
        basename=$(basename $filename)
        . /home/s3094723/SEMEVAL/affecthor/utils/format_arabic_lexicon.sh $filename ${AR_LEX_ODIR}/$basename.clean
done


