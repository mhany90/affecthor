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
if [ "$lexfile" = "Arabic_Emoticon_Lexicon.txt" ]; then
    field_ar=1
    field_em=3
    data_start=105

elif [ "$lexfile" = "Arabic_Hashtag_Lexicon_dialectal.txt" ] || [ "$lexfile" = "Arabic_Hashtag_Lexicon.txt" ]; then
    field_ar=1
    field_em=3
    data_start=106

elif [ "$lexfile" = "bingliu_ar.txt" ]; then
    field_ar=2
    field_em=4
    data_start=87

elif [ "$lexfile" = "MPQA_ar.txt" ]; then
    field_ar=14
    field_em=12
    data_start=83
    field_sep="[[[:space:]=]"

elif [ "$lexfile" = "nrc_emotion.txt" ]; then
    field_ar=4
    field_em=2
    data_start=1

elif [ "$lexfile" = "NRC-HS-unigrams-pmilexicon_ar.txt" ] || [ "$lexfile" = "S140-unigrams-pmilexicon_ar.txt" ]; then
    field_ar=2
    field_em=3
    data_start=90
fi

# extract and output
awk -v awk_ar=$field_ar -v awk_em=$field_em -v awk_s=$data_start -F $field_sep '{
	if(NR >= awk_s) {
		if(NF > 1) {
			print $awk_ar "\t" $awk_em
		}
	}
}' $IFILE > $OFILE

# extra formatting steps for bad shaped lexicons...
if [ "$lexfile" = "MPQA_ar.txt" ]; then
    grep "$(printf '\t')positive\|$(printf '\t')negative\|$(printf '\t')neutral\|$(printf '\t')both" $OFILE > $OFILE.clean
    sed -i '1s/^/[ar]\t[priorpolarity]\n/' $OFILE.clean
    rm -f $OFILE
    mv $OFILE.clean $OFILE
    rm -f $OFILE.clean
elif [ "$lexfile" = "nrc_emotion.txt" ]; then
    sort $OFILE | uniq > $OFILE.clean
    sed -i '1s/^/[Arabic translation]\t[Emotion Indicator]\n/' $OFILE.clean
    rm -f $OFILE
    mv $OFILE.clean $OFILE
    rm -f $OFILE.clean
fi

