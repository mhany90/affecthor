#!/bin/bash
# this scripts extracts from and formats the Arabic lexicons

# input lexicon
IFILE=$1

# output lexicon
OFILE=$2

# determine the file name
lexfile=$(basename $IFILE)

# field of the Arabic word
field_ar=0

# field of the emotion value
field_em=0

# line number where data starts
data_start=0

# field separator
field_sep="[\t]"

# determine values according to the file format
if [ "$lexfile" = "Arabic_Emoticon_Lexicon.txt" ] || [ "$lexfile" = "Arabic_Hashtag_Lexicon_dialectal.txt" ] || \
       [ "$lexfile" = "Arabic_Hashtag_Lexicon.txt" ]; then
    field_ar=1
    field_em=3
    data_start=105

elif [ "$lexfile" = "bingliu_ar.txt" ]; then
    field_ar=2
    field_em=4
    data_start=87

elif [ "$lexfile" = "MPQA_ar.txt" ]; then
    field_ar=14
    field_em=12
    data_start=40
    field_sep="[[[:space:]=]"

elif [ "$lexfile" = "nrc_emotion_ar.txt" ]; then
    field_ar=4
    field_em=3
    data_start=39

elif [ "$lexfile" = "NRC-HS-unigrams-pmilexicon_ar.txt" ] || [ "$lexfile" = "S140-unigrams-pmilexicon_ar.txt" ]; then
    field_ar=2
    field_em=3
    data_start=32

# extract and output
awk -v awk_ar=$field_ar -v awk_em=$field_em -v awk_s=$data_start -F $field_sep '{
	if(NR >= awk_s) {
		if(NF > 1) {
			print $awk_ar "\t" $awk_em
		}
	}
}' $IFILE > $OFILE

