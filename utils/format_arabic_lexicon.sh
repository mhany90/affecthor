#!/bin/bash
# this scripts extracts from and formats the Arabic lexicons


# input lexicon
IFILE=$1

# output lexicon
OFILE=$2

# determine the file name
lexfile=$(basename $IFILE)

# ordered fields of interest where the Arabic word comes first
fields="1 2"

# header with attribute names
header="word\tscore"

# field separator
field_sep="[\t]"


# determine values according to the file format
if [ "$lexfile" = "Arabic_Emoticon_Lexicon.txt" ]; then
    sed '1,105d' "${IFILE}" > "${IFILE}.tmp"
    fields="1 4 5"
    header="word\t[Buckwalter]\t[Sentiment Score]\tEmoLex-posCount\tEmoLex-negCount"
    field_sep="[\t]"


elif [ "$lexfile" = "Arabic_Hashtag_Lexicon.txt" ]; then
    sed '1,106d' "${IFILE}" > "${IFILE}.tmp"
    fields="1 4 5"
    header="word\t[Buckwalter]\t[Sentiment Score]\tHash-posCount\tHash-negCount"
    field_sep="[\t]"


elif [ "$lexfile" = "Arabic_Hashtag_Lexicon_dialectal.txt" ]; then
    sed '1,106d' "${IFILE}" > "${IFILE}.tmp"
    fields="1 4 5"
    header="word\t[Buckwalter]\t[Sentiment Score]\tHash-dial-posCount\tHash-dial-negCount"
    field_sep="[\t]"


elif [ "$lexfile" = "bingliu_ar.txt" ]; then
    sed '1,87d' "${IFILE}" > "${IFILE}.tmp"
    fields="2 4"
    header="[English Term]\tword\t[Buckwalter]\tBingLiu-sent"
    field_sep="[\t]"


elif [ "$lexfile" = "MPQA_ar.txt" ]; then
    sed '1,82d' "${IFILE}" > "${IFILE}.tmp"
    fields="14 12"
    header="[t]=[tt]\ [l]=[ll]\ [w]=[ww]\ [p]=[pp]\ [s]=[ss]\ [x]=mpqa-polarity\ [a]=word\ [b]=[bb]"
    field_sep="[[:space:]=]"


elif [ "$lexfile" = "nrc_emotion_ar.txt" ]; then
    sed '1,85d' "${IFILE}" > "${IFILE}.tmp2"
    # combine entries with the same word but different emotion on a single line
    awk 'BEGIN { FS="\t" } { printf "%s", $0; if (NR % 10 == 0) print ""; else printf "\t" }' "${IFILE}.tmp2" > "${IFILE}.tmp"
    sed -i 's/ /_/g' "${IFILE}.tmp"
    fields="4 3 8 13 18 23 28 33 38 43 48"
    # build and insert header
    atts=("NRC-10-anger" "NRC-10-anticipation" "NRC-10-disgust" "NRC-10-fear" "NRC-10-joy"
          "NRC-10-negative" "NRC-10-positive" "NRC-10-sadness" "NRC-10-surprise" "NRC-10-trust")
    for att in ${atts[@]}; do
        if [ "$header" != "word\tscore" ]; then
            header="${header}\t"
        else
            header=""
        fi
        header="${header}[English Term]\t[Emotion]\t${att}\tword\t[Buckwalter]"
    done
    field_sep="[\t]"


elif [ "$lexfile" = "NRC-HS-unigrams-pmilexicon_ar.txt" ]; then
    sed '1,90d' "${IFILE}" > "${IFILE}.tmp"
    fields="2 4 5"
    header="[English Term]\tword\t[Sentiment Score]\tNRC-Hash-posCount\tNRC-Hash-negCount"
    field_sep="[\t]"


elif [ "$lexfile" = "S140-unigrams-pmilexicon_ar.txt" ]; then
    sed '1,90d' "${IFILE}" > "${IFILE}.tmp"
    fields="2 4 5"
    header="[English Term]\tword\t[Sentiment Score]\tS140-posCount\tS140-negCount"
    field_sep="[\t]"

elif [ "$lexfile" = "AFINN-emoticon-8.txt" ]; then
    cp "${IFILE}" "${IFILE}.tmp"
    fields="1 2"
    header="word\tafinn-emoticon-score"
    field_sep="[\t]"
fi

# extract and output
sed -i "1s/^/${header}\n/" "${IFILE}.tmp"
awk -v awk_fields="$fields" -F $field_sep '
BEGIN { split(awk_fields, A, / /) }
{
  for ( i=1; i<length(A); i++ ) printf $(A[i]) "\t"; print $(A[length(A)])
}' "${IFILE}.tmp" > $OFILE

rm -f $IFILE.tmp
rm -f $IFILE.tmp2


# extra formatting steps for bad shaped lexicons...
if [ "$lexfile" = "MPQA_ar.txt" ]; then
    grep "$(printf '\t')positive\|$(printf '\t')negative\|$(printf '\t')neutral\|$(printf '\t')both" $OFILE > $OFILE.clean
    sed -i '1s/^/word\tmpqa-polarity\n/' $OFILE.clean
    rm -f $OFILE
    mv $OFILE.clean $OFILE
    rm -f $OFILE.clean

elif [ "$lexfile" = "nrc_emotion_ar.txt" ]; then
    python3 ${UTILSDIR}/format_arabic_lexicon_nrc.py $OFILE $OFILE.clean
    sed -i 's/_/ /g' $OFILE.clean
    rm -f $OFILE
    mv $OFILE.clean $OFILE
    rm -f $OFILE.clean
fi

